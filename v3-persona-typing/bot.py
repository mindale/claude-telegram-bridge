#!/usr/bin/env python3
"""
Мост между Telegram и Claude Code (запущенным локально на сервере через
вашу подписку Claude, БЕЗ Anthropic API).

Поддерживает: текст, голосовые сообщения (транскрибируются локально через
faster-whisper), фото (передаются Claude Code как файл в рабочей директории).

Каждый Telegram-чат получает свою рабочую директорию -> Claude Code хранит
для неё отдельную историю сессии, так что разговор не путается между чатами.

Запуск:
    python3 bot.py

Конфигурация — через переменные окружения (см. .env.example).
"""

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import uuid
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.enums import ChatAction, ParseMode
from dotenv import load_dotenv
import telegramify_markdown

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("claude-bridge")

# ---------- Конфигурация ----------
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").split(",") if x.strip()
}
if not ALLOWED_USER_IDS:
    raise SystemExit("Задайте ALLOWED_USER_IDS в .env — иначе бот открыт всем в интернете.")

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
BASE_DIR = Path(os.environ.get("SESSIONS_DIR", "./sessions")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_FILE = BASE_DIR / "_sessions.json"

CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "300"))

DEFAULT_PERSONA = (
    "Ты общаешься со мной в Telegram, как close friend, а не как ассистент "
    "техподдержки. Пиши живо и по-человечески: обычные разговорные фразы, "
    "можно юмор и эмоции, без канцелярита. Не форматируй ответы как markdown-"
    "документ — без заголовков (#), без жирного через **, без длинных "
    "списков и разделителей. Пиши абзацами, как в обычной переписке. Если "
    "нужно перечислить пункты — просто через тире на новой строке, коротко."
)
# Системная "личность" Claude в чате. Переопределяется через CLAUDE_PERSONA в .env.
CLAUDE_PERSONA = os.environ.get("CLAUDE_PERSONA", DEFAULT_PERSONA)

# Доп. флаги для claude, например: "--dangerously-skip-permissions" или
# "--allowedTools Read,Grep,Glob". Смотрите README про риски.
CLAUDE_EXTRA_ARGS = shlex.split(os.environ.get("CLAUDE_EXTRA_ARGS", ""))
if CLAUDE_PERSONA:
    CLAUDE_EXTRA_ARGS += ["--append-system-prompt", CLAUDE_PERSONA]

TYPING_REFRESH_SECONDS = 4

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ru")

TELEGRAM_MAX_LEN = 4000

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ---------- Хранилище session_id по чатам ----------

def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text())
    return {}


def save_sessions(data: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def chat_dir(chat_id: int) -> Path:
    d = BASE_DIR / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Ленивая загрузка Whisper (только при первом голосовом) ----------
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info("Загружаю модель Whisper (%s)...", WHISPER_MODEL_SIZE)
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_path: Path) -> str:
    model = get_whisper_model()
    segments, _info = model.transcribe(str(audio_path), language=WHISPER_LANGUAGE)
    return " ".join(seg.text.strip() for seg in segments).strip()


# ---------- Вызов Claude Code ----------

async def run_claude(chat_id: int, prompt: str) -> str:
    sessions = load_sessions()
    key = str(chat_id)
    session_id = sessions.get(key)

    cwd = chat_dir(chat_id)
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json"]
    if session_id:
        cmd += ["--resume", session_id]
    cmd += CLAUDE_EXTRA_ARGS

    log.info("chat=%s -> claude %s", chat_id, " ".join(shlex.quote(c) for c in cmd[:3]))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CLAUDE_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return "⏱️ Claude Code не ответил вовремя (таймаут). Попробуйте ещё раз или упростите запрос."

    if proc.returncode != 0:
        log.error("claude stderr: %s", stderr.decode(errors="replace"))
        return f"⚠️ Claude Code завершился с ошибкой:\n```\n{stderr.decode(errors='replace')[-1500:]}\n```"

    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return stdout.decode(errors="replace") or "⚠️ Пустой ответ от Claude Code."

    new_session_id = data.get("session_id")
    if new_session_id:
        sessions[key] = new_session_id
        save_sessions(sessions)

    return data.get("result") or "⚠️ Claude Code не вернул текст ответа."


def split_markdown_blocks(text: str, max_len: int) -> list[str]:
    """Режет markdown-текст на куски, никогда не разрывая блок кода (```...```)
    посередине — иначе после конвертации в MarkdownV2 получаются куски с
    незакрытым тройным бэктиком, и Telegram отклоняет всё сообщение целиком."""
    lines = text.split("\n")
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        if buf:
            blocks.append("\n".join(buf))
            buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                # начинается блок кода — если накопился обычный текст, отделяем его
                flush()
                in_fence = True
                buf.append(line)
            else:
                # блок кода закрывается — фиксируем его целиком одним куском
                buf.append(line)
                flush()
                in_fence = False
        elif in_fence:
            buf.append(line)
        elif stripped == "" and buf:
            flush()
        else:
            buf.append(line)
    flush()

    # Собираем блоки в чанки под лимит длины, не разрезая блок посередине
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(block) <= max_len:
                current = block
            else:
                # Единичный блок сам больше лимита (гигантский код) — режем построчно как крайний случай
                for i in range(0, len(block), max_len):
                    chunks.append(block[i : i + max_len])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [""]


async def send_long(message: Message, text: str) -> None:
    """Отправляет ответ Claude, конвертируя markdown в формат, который
    Telegram реально отрисовывает (жирный, списки, код), вместо того чтобы
    показывать сырые звёздочки и решётки. Режет по границам блоков, чтобы не
    разрывать код посередине, и откатывает в обычный текст только тот кусок,
    который не удалось отправить — а не весь ответ."""
    for block in split_markdown_blocks(text, TELEGRAM_MAX_LEN):
        try:
            converted = telegramify_markdown.markdownify(block)
            await message.answer(converted, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            log.exception("Не удалось отправить кусок как MarkdownV2, отправляю обычным текстом")
            await message.answer(block)


def check_access(message: Message) -> bool:
    return message.from_user is not None and message.from_user.id in ALLOWED_USER_IDS


class TypingIndicator:
    """Держит статус "печатает..." живым, пока Claude думает.

    Telegram гасит индикатор через ~5 секунд, поэтому просто один раз
    отправить ChatAction.TYPING перед долгим вызовом claude недостаточно —
    нужно обновлять его в фоне, пока ответ не готов.
    Использование: async with TypingIndicator(chat_id): ...долгая работа...
    """

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self._task: asyncio.Task | None = None

    async def _loop(self):
        try:
            while True:
                await bot.send_chat_action(self.chat_id, ChatAction.TYPING)
                await asyncio.sleep(TYPING_REFRESH_SECONDS)
        except asyncio.CancelledError:
            pass

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


# ---------- Хендлеры ----------

@dp.message(Command("start"))
async def cmd_start(message: Message):
    if not check_access(message):
        return
    await message.answer(
        "Привет! Это мост к Claude Code на сервере.\n"
        "Пишите текстом, присылайте голосовые или фото — отвечу через Claude.\n"
        "/reset — начать разговор заново (сбросить сессию)."
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    if not check_access(message):
        return
    sessions = load_sessions()
    sessions.pop(str(message.chat.id), None)
    save_sessions(sessions)
    await message.answer("Сессия сброшена. Начинаем разговор с чистого листа.")


@dp.message(F.text)
async def on_text(message: Message):
    if not check_access(message):
        await message.answer("⛔ У вас нет доступа к этому боту.")
        return
    async with TypingIndicator(message.chat.id):
        reply = await run_claude(message.chat.id, message.text)
    await send_long(message, reply)


@dp.message(F.voice)
async def on_voice(message: Message):
    if not check_access(message):
        return

    async with TypingIndicator(message.chat.id):
        cwd = chat_dir(message.chat.id)
        ogg_path = cwd / f"voice_{uuid.uuid4().hex}.ogg"
        wav_path = ogg_path.with_suffix(".wav")

        file = await bot.get_file(message.voice.file_id)
        await bot.download_file(file.file_path, destination=ogg_path)

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ogg_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
        )

        text = await asyncio.to_thread(transcribe, wav_path)
        ogg_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        if not text:
            await message.answer("Не удалось распознать голосовое сообщение.")
            return

        await message.answer(f"🎤 Распознано: {text}")
        reply = await run_claude(message.chat.id, text)
    await send_long(message, reply)


@dp.message(F.photo)
async def on_photo(message: Message):
    if not check_access(message):
        return
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    cwd = chat_dir(message.chat.id)
    photo = message.photo[-1]
    img_name = f"image_{uuid.uuid4().hex}.jpg"
    img_path = cwd / img_name

    file = await bot.get_file(photo.file_id)
    await bot.download_file(file.file_path, destination=img_path)

    caption = message.caption or "Посмотри на это изображение и опиши/прокомментируй его."
    prompt = f"{caption}\n\n(файл изображения лежит в рабочей директории: {img_name})"

    reply = await run_claude(message.chat.id, prompt)
    await send_long(message, reply)


async def main():
    log.info("Бот запущен. Разрешённые пользователи: %s", ALLOWED_USER_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

