#!/usr/bin/env python3
"""
Мост между Telegram и Claude Code (запущенным локально на сервере через
вашу подписку Claude, БЕЗ Anthropic API).

Поддерживает: текст, голосовые сообщения (транскрибируются локально через
faster-whisper), фото (передаются Claude Code как файл в рабочей директории).

Голос — двусторонне и зеркально: если вы прислали голосовое, ответ Claude
озвучивается голосом (локально, через Silero TTS) и отправляется обратно как
голосовое сообщение; на обычный текст ответ приходит текстом, как раньше.

Внутри одного Telegram-чата можно вести НЕСКОЛЬКО независимых веток
разговора ("чатов") — у каждой своя история и свой стиль/скиллы:

    /newchat <имя>       — создать новую ветку и переключиться на неё
    /chats               — список веток, отметка активной
    /switch <имя>        — переключиться на существующую ветку
    /rename <имя>        — переименовать текущую ветку
    /delchat <имя>       — удалить ветку (кроме активной)
    /reset               — сбросить историю текущей ветки (сессию Claude)

    /style <текст>       — задать свой стиль общения для текущей ветки
    /resetstyle          — вернуть стиль по умолчанию (CLAUDE_PERSONA)
    /skills              — список доступных скиллов (из skills.json)
    /addskill <имя>      — подключить скилл к текущей ветке
    /removeskill <имя>   — отключить скилл
    /status              — показать активную ветку, стиль, скиллы, режим голосовых

    /voice [имя]         — показать/сменить голос озвучки для текущей ветки
                            (aidar, baya, kseniya, xenia, eugene, random)
    /voicemode [режим]   — assistant (по умолчанию, зеркально) или transcribe
                            (только расшифровка голосовых, Claude не вызывается);
                            без аргумента переключает

Запуск:
    python3 bot.py

Конфигурация — через переменные окружения (см. .env.example) и skills.json.
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
from aiogram.filters import Command, CommandObject
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
STATE_FILE = BASE_DIR / "_state.json"
SKILLS_FILE = Path(os.environ.get("SKILLS_FILE", "./skills.json")).resolve()

CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "300"))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT_SECONDS", "60"))
VOICE_DOWNLOAD_TIMEOUT = int(os.environ.get("VOICE_DOWNLOAD_TIMEOUT_SECONDS", "60"))

DEFAULT_PERSONA = (
    "Ты общаешься со мной в Telegram, как close friend, а не как ассистент "
    "техподдержки. Пиши живо и по-человечески: обычные разговорные фразы, "
    "можно юмор и эмоции, без канцелярита. Не форматируй ответы как markdown-"
    "документ — без заголовков (#), без жирного через **, без длинных "
    "списков и разделителей. Пиши абзацами, как в обычной переписке. Если "
    "нужно перечислить пункты — просто через тире на новой строке, коротко."
)
# Стиль по умолчанию для новых веток. Переопределяется через CLAUDE_PERSONA в .env.
CLAUDE_PERSONA = os.environ.get("CLAUDE_PERSONA", DEFAULT_PERSONA)

# Доп. флаги для claude (НЕ включают стиль — он собирается отдельно на каждый
# вызов, т.к. зависит от активной ветки). Пример: "--dangerously-skip-permissions"
# или "--allowedTools Read,Grep,Glob". Смотрите README про риски.
BASE_EXTRA_ARGS = shlex.split(os.environ.get("CLAUDE_EXTRA_ARGS", ""))

TYPING_REFRESH_SECONDS = 4
DEFAULT_THREAD = "general"

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ru")

# ---------- TTS (озвучка ответов через Silero) ----------
TTS_ENABLED = os.environ.get("TTS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
# Голоса модели v4_ru. Дефолтный — из .env, конкретная ветка может выбрать
# свой через /voice <имя>.
TTS_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "aidar")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "48000"))  # 8000 / 24000 / 48000
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cpu")
# Сколько символов ответа озвучивать за один раз. У Silero есть внутренний
# лимит на длину строки за один вызов (около 1000 символов) — режем текст на
# куски по границам предложений и склеиваем звук, но всё вместе не длиннее
# TTS_MAX_CHARS, иначе голосовое получится неприлично длинным.
TTS_CHUNK_CHARS = 700
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "4000"))
TTS_TIMEOUT = int(os.environ.get("TTS_TIMEOUT_SECONDS", "120"))

TELEGRAM_MAX_LEN = 4000

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ---------- Скиллы (пресеты системного промпта) ----------

DEFAULT_SKILLS = {
    "программист": (
        "Когда я спрашиваю про код — отвечай как опытный инженер: конкретно, "
        "с примерами кода, без воды. Указывай на подводные камни."
    ),
    "коротко": "Отвечай максимально коротко — 1-3 предложения, без вступлений.",
    "училка": (
        "Объясняй как учитель терпеливому ученику: пошагово, простыми словами, "
        "с примерами, не стесняйся переспрашивать, понятно ли."
    ),
    "юморист": "Добавляй лёгкий юмор и иронию в ответы, где это уместно.",
}


def load_skills() -> dict:
    if SKILLS_FILE.exists():
        try:
            return json.loads(SKILLS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.exception("Не удалось разобрать %s, использую скиллы по умолчанию", SKILLS_FILE)
    SKILLS_FILE.write_text(json.dumps(DEFAULT_SKILLS, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(DEFAULT_SKILLS)


SKILLS = load_skills()

# ---------- Состояние: ветки чата, стили, скиллы, session_id ----------
#
# Структура STATE_FILE:
# {
#   "<chat_id>": {
#     "active": "general",
#     "threads": {
#       "general": {"session_id": null, "persona": null, "skills": []},
#       "работа":  {"session_id": "...", "persona": "...", "skills": ["программист"]}
#     }
#   }
# }


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_chat_state(state: dict, chat_id: int) -> dict:
    key = str(chat_id)
    if key not in state:
        state[key] = {"active": DEFAULT_THREAD, "threads": {DEFAULT_THREAD: new_thread()}}
    return state[key]


def new_thread() -> dict:
    return {
        "session_id": None,
        "persona": None,
        "skills": [],
        # "assistant" — голосовое расшифровывается и уходит Claude, ответ
        # приходит голосом (зеркально). "transcribe" — только расшифровка,
        # Claude не вызывается (переключается командой /voicemode).
        "voice_mode": "assistant",
        # None — использовать глобальный TTS_SPEAKER из .env; иначе — свой
        # голос для этой ветки (задаётся командой /voice <имя>).
        "tts_speaker": None,
    }


def get_active_thread(chat_id: int) -> tuple[dict, str, dict]:
    """Возвращает (state, thread_name, thread_dict) для активной ветки чата."""
    state = load_state()
    chat_state = get_chat_state(state, chat_id)
    name = chat_state["active"]
    if name not in chat_state["threads"]:
        name = DEFAULT_THREAD
        chat_state["active"] = name
        chat_state["threads"].setdefault(name, new_thread())
    return state, name, chat_state["threads"][name]


def thread_dir(chat_id: int, thread_name: str) -> Path:
    d = BASE_DIR / str(chat_id) / thread_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_system_prompt(thread: dict) -> str:
    parts = [thread.get("persona") or CLAUDE_PERSONA]
    for skill_name in thread.get("skills", []):
        text = SKILLS.get(skill_name)
        if text:
            parts.append(text)
    return "\n\n".join(p for p in parts if p)


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


# ---------- Ленивая загрузка Silero TTS (только при первой озвучке) ----------
_tts_model = None


def get_tts_model():
    """Загружает модель Silero TTS (v4_ru) через torch.hub. Модель кешируется
    на диске (~/.cache/torch/hub) после первой загрузки — в дальнейшем
    интернет для неё не нужен, только при самом первом запуске."""
    global _tts_model
    if _tts_model is None:
        import torch
        log.info("Загружаю модель Silero TTS (v4_ru)...")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        model.to(TTS_DEVICE)
        _tts_model = model
    return _tts_model


def split_text_for_tts(text: str, max_len: int) -> list[str]:
    """Режет текст на куски по границам предложений так, чтобы каждый кусок
    укладывался в лимит длины одного вызова Silero (иначе модель бросает
    ValueError на длинных строках)."""
    import re

    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_len:
                current = sentence
            else:
                # Одно предложение само по себе длиннее лимита — режем жёстко.
                for i in range(0, len(sentence), max_len):
                    chunks.append(sentence[i : i + max_len])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [text[:max_len]]


def synthesize_speech(text: str, wav_path: Path, speaker: str) -> None:
    """Синхронно синтезирует речь и пишет .wav. Вызывать только через
    asyncio.to_thread — сама модель синхронная и не должна выполняться
    прямо в event loop."""
    import torch
    import soundfile as sf

    model = get_tts_model()
    chunks = split_text_for_tts(text, TTS_CHUNK_CHARS)
    parts = [
        model.apply_tts(text=chunk, speaker=speaker, sample_rate=TTS_SAMPLE_RATE)
        for chunk in chunks
    ]
    audio = torch.cat(parts) if len(parts) > 1 else parts[0]
    sf.write(str(wav_path), audio.numpy(), TTS_SAMPLE_RATE)


def convert_wav_to_voice_ogg(wav_path: Path, ogg_path: Path) -> None:
    """Синхронно конвертирует .wav в .ogg/Opus — формат, который Telegram
    принимает для голосовых сообщений (send_voice). Вызывать только через
    asyncio.to_thread."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "libopus", "-b:a", "48k", "-vbr", "on", str(ogg_path)],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
    )


async def send_reply(message: Message, text: str, cwd: Path, as_voice: bool, speaker: str | None = None) -> None:
    """Отправляет ответ Claude пользователю: текстом (как раньше) или, если
    as_voice=True и включён TTS, голосовым сообщением. speaker — голос для
    этого конкретного ответа (обычно из настройки ветки); None — берётся
    глобальный TTS_SPEAKER из .env. При любой ошибке синтеза/конвертации
    молча откатывается на текст — пользователь в любом случае получает
    ответ."""
    if not (as_voice and TTS_ENABLED):
        await send_long(message, text)
        return

    speaker = speaker or TTS_SPEAKER

    tts_text = text
    truncated = False
    if len(tts_text) > TTS_MAX_CHARS:
        tts_text = tts_text[:TTS_MAX_CHARS]
        truncated = True

    wav_path: Path | None = None
    ogg_path: Path | None = None
    try:
        wav_path = cwd / f"tts_{uuid.uuid4().hex}.wav"
        ogg_path = wav_path.with_suffix(".ogg")

        log.info("chat=%s: синтезирую речь (Silero TTS, голос=%s)", message.chat.id, speaker)
        await asyncio.wait_for(
            asyncio.to_thread(synthesize_speech, tts_text, wav_path, speaker), timeout=TTS_TIMEOUT
        )
        await asyncio.to_thread(convert_wav_to_voice_ogg, wav_path, ogg_path)

        await message.answer_voice(FSInputFile(ogg_path))
        if truncated:
            await message.answer(
                f"✂️ Ответ длиннее {TTS_MAX_CHARS} символов, озвучено только начало. Текст целиком:"
            )
            await send_long(message, text)
    except Exception:
        log.exception("chat=%s: не удалось озвучить ответ, отправляю текстом", message.chat.id)
        await send_long(message, text)
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
        if ogg_path is not None:
            ogg_path.unlink(missing_ok=True)


# ---------- Вызов Claude Code ----------

async def run_claude(chat_id: int, prompt: str) -> str:
    state, thread_name, thread = get_active_thread(chat_id)
    session_id = thread.get("session_id")

    cwd = thread_dir(chat_id, thread_name)
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json"]
    if session_id:
        cmd += ["--resume", session_id]
    cmd += BASE_EXTRA_ARGS

    system_prompt = build_system_prompt(thread)
    if system_prompt:
        cmd += ["--append-system-prompt", system_prompt]

    log.info("chat=%s thread=%s -> claude %s", chat_id, thread_name, " ".join(shlex.quote(c) for c in cmd[:3]))

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
        # перечитываем состояние на случай параллельных изменений и пишем в ту же ветку
        state = load_state()
        chat_state = get_chat_state(state, chat_id)
        chat_state["threads"].setdefault(thread_name, new_thread())["session_id"] = new_session_id
        save_state(state)

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
                flush()
                in_fence = True
                buf.append(line)
            else:
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
                for i in range(0, len(block), max_len):
                    chunks.append(block[i : i + max_len])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [""]


async def send_long(message: Message, text: str) -> None:
    """Отправляет ответ Claude, конвертируя markdown в формат, который
    Telegram реально отрисовывает. Режет по границам блоков, чтобы не
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
    """Держит статус-индикатор Telegram живым, пока Claude думает.

    По умолчанию — "печатает...". Для голосового диалога (когда ответ
    придёт голосовым сообщением) передайте action=ChatAction.RECORD_VOICE —
    пользователь увидит "записывает голосовое...", что честнее отражает,
    что сейчас происходит."""

    def __init__(self, chat_id: int, action: ChatAction = ChatAction.TYPING):
        self.chat_id = chat_id
        self.action = action
        self._task: asyncio.Task | None = None

    async def _loop(self):
        try:
            while True:
                await bot.send_chat_action(self.chat_id, self.action)
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


# ---------- Хендлеры: базовые ----------

@dp.message(Command("start"))
async def cmd_start(message: Message):
    if not check_access(message):
        return
    await message.answer(
        "Привет! Это мост к Claude Code на сервере.\n"
        "Пишите текстом, присылайте голосовые или фото — отвечу через Claude.\n\n"
        "Ветки разговора: /newchat, /chats, /switch, /rename, /delchat\n"
        "Стиль и скиллы: /style, /resetstyle, /skills, /addskill, /removeskill\n"
        "Голос: /voice — сменить голос озвучки. /voicemode — вкл/выкл ответ Claude "
        "на голосовые (можно оставить только расшифровку).\n"
        "/reset — сбросить историю текущей ветки. /status — что сейчас активно."
    )


@dp.message(Command("status"))
async def cmd_status(message: Message):
    if not check_access(message):
        return
    _, name, thread = get_active_thread(message.chat.id)
    persona = thread.get("persona") or f"(по умолчанию) {CLAUDE_PERSONA}"
    skills = ", ".join(thread.get("skills", [])) or "нет"
    has_history = "есть" if thread.get("session_id") else "нет (начнётся заново)"
    voice_mode = thread.get("voice_mode", "assistant")
    voice_mode_label = (
        "только расшифровка, Claude не вызывается" if voice_mode == "transcribe"
        else "обычный — расшифровка + ответ Claude голосом"
    )
    tts_speaker = thread.get("tts_speaker")
    voice_label = tts_speaker or f"{TTS_SPEAKER} (по умолчанию)"
    await message.answer(
        f"Активная ветка: {name}\n"
        f"История разговора: {has_history}\n"
        f"Скиллы: {skills}\n"
        f"Стиль: {persona}\n"
        f"Режим голосовых: {voice_mode_label}\n"
        f"Голос TTS: {voice_label}"
    )


# ---------- Хендлеры: ветки ("чаты") ----------

@dp.message(Command("newchat"))
async def cmd_newchat(message: Message, command: CommandObject):
    if not check_access(message):
        return
    name = (command.args or "").strip()
    if not name:
        await message.answer("Использование: /newchat <имя>, например /newchat работа")
        return
    state = load_state()
    chat_state = get_chat_state(state, message.chat.id)
    if name in chat_state["threads"]:
        await message.answer(f"Ветка «{name}» уже есть. Переключаюсь на неё.")
    else:
        chat_state["threads"][name] = new_thread()
        await message.answer(f"Создал ветку «{name}» и переключился на неё. История разговора — с чистого листа.")
    chat_state["active"] = name
    save_state(state)


@dp.message(Command("chats"))
async def cmd_chats(message: Message):
    if not check_access(message):
        return
    state = load_state()
    chat_state = get_chat_state(state, message.chat.id)
    save_state(state)
    lines = []
    for name in chat_state["threads"]:
        marker = "→ " if name == chat_state["active"] else "  "
        lines.append(f"{marker}{name}")
    await message.answer("Ваши ветки:\n" + "\n".join(lines))


@dp.message(Command("switch"))
async def cmd_switch(message: Message, command: CommandObject):
    if not check_access(message):
        return
    name = (command.args or "").strip()
    state = load_state()
    chat_state = get_chat_state(state, message.chat.id)
    if not name:
        await message.answer("Использование: /switch <имя>. Список веток — /chats")
        return
    if name not in chat_state["threads"]:
        await message.answer(f"Нет такой ветки: «{name}». Список веток — /chats, новая — /newchat {name}")
        return
    chat_state["active"] = name
    save_state(state)
    await message.answer(f"Переключился на ветку «{name}».")


@dp.message(Command("rename"))
async def cmd_rename(message: Message, command: CommandObject):
    if not check_access(message):
        return
    new_name = (command.args or "").strip()
    if not new_name:
        await message.answer("Использование: /rename <новое имя>")
        return
    state = load_state()
    chat_state = get_chat_state(state, message.chat.id)
    old_name = chat_state["active"]
    if new_name in chat_state["threads"]:
        await message.answer(f"Ветка «{new_name}» уже существует.")
        return
    chat_state["threads"][new_name] = chat_state["threads"].pop(old_name)
    chat_state["active"] = new_name
    save_state(state)
    # переносим и рабочую директорию, чтобы Claude Code не потерял файлы/сессию на диске
    old_dir = BASE_DIR / str(message.chat.id) / old_name
    new_dir = BASE_DIR / str(message.chat.id) / new_name
    if old_dir.exists() and not new_dir.exists():
        old_dir.rename(new_dir)
    await message.answer(f"Ветка «{old_name}» переименована в «{new_name}».")


@dp.message(Command("delchat"))
async def cmd_delchat(message: Message, command: CommandObject):
    if not check_access(message):
        return
    name = (command.args or "").strip()
    state = load_state()
    chat_state = get_chat_state(state, message.chat.id)
    if not name:
        await message.answer("Использование: /delchat <имя>")
        return
    if name == chat_state["active"]:
        await message.answer("Нельзя удалить активную ветку — сначала переключитесь на другую через /switch.")
        return
    if name not in chat_state["threads"]:
        await message.answer(f"Нет такой ветки: «{name}».")
        return
    del chat_state["threads"][name]
    save_state(state)
    await message.answer(f"Ветка «{name}» удалена.")


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    if not check_access(message):
        return
    state = load_state()
    _, name, thread = get_active_thread(message.chat.id)
    thread["session_id"] = None
    save_state(state)
    await message.answer(f"История ветки «{name}» сброшена. Начинаем разговор с чистого листа.")


# ---------- Хендлеры: стиль и скиллы ----------

@dp.message(Command("style"))
async def cmd_style(message: Message, command: CommandObject):
    if not check_access(message):
        return
    text = (command.args or "").strip()
    state = load_state()
    _, name, thread = get_active_thread(message.chat.id)
    if not text:
        await message.answer(
            "Использование: /style <описание стиля>, например:\n"
            "/style Отвечай кратко и по делу, без эмоций, как технический консультант."
        )
        return
    thread["persona"] = text
    save_state(state)
    await message.answer(f"Стиль ветки «{name}» обновлён.")


@dp.message(Command("resetstyle"))
async def cmd_resetstyle(message: Message):
    if not check_access(message):
        return
    state = load_state()
    _, name, thread = get_active_thread(message.chat.id)
    thread["persona"] = None
    save_state(state)
    await message.answer(f"Стиль ветки «{name}» сброшен на стандартный.")


@dp.message(Command("skills"))
async def cmd_skills(message: Message):
    if not check_access(message):
        return
    global SKILLS
    SKILLS = load_skills()  # подхватываем правки skills.json без перезапуска бота
    _, _, thread = get_active_thread(message.chat.id)
    active = set(thread.get("skills", []))
    lines = []
    for name, text in SKILLS.items():
        marker = "✅" if name in active else "▫️"
        lines.append(f"{marker} {name} — {text[:60]}{'…' if len(text) > 60 else ''}")
    lines.append("")
    lines.append("Подключить: /addskill <имя>. Отключить: /removeskill <имя>.")
    lines.append(f"Свои скиллы можно добавить прямо в файл {SKILLS_FILE.name} на сервере.")
    await message.answer("\n".join(lines))


@dp.message(Command("addskill"))
async def cmd_addskill(message: Message, command: CommandObject):
    if not check_access(message):
        return
    global SKILLS
    SKILLS = load_skills()
    name = (command.args or "").strip()
    if name not in SKILLS:
        await message.answer(f"Нет скилла «{name}». Список — /skills")
        return
    state = load_state()
    _, thread_name, thread = get_active_thread(message.chat.id)
    if name not in thread["skills"]:
        thread["skills"].append(name)
        save_state(state)
    await message.answer(f"Скилл «{name}» подключён к ветке «{thread_name}».")


@dp.message(Command("removeskill"))
async def cmd_removeskill(message: Message, command: CommandObject):
    if not check_access(message):
        return
    name = (command.args or "").strip()
    state = load_state()
    _, thread_name, thread = get_active_thread(message.chat.id)
    if name in thread["skills"]:
        thread["skills"].remove(name)
        save_state(state)
        await message.answer(f"Скилл «{name}» отключён от ветки «{thread_name}».")
    else:
        await message.answer(f"Скилл «{name}» и не был подключён к этой ветке.")


# ---------- Хендлеры: голос ----------

@dp.message(Command("voice"))
async def cmd_voice(message: Message, command: CommandObject):
    if not check_access(message):
        return
    arg = (command.args or "").strip().lower()
    state = load_state()
    _, thread_name, thread = get_active_thread(message.chat.id)
    current = thread.get("tts_speaker")

    if not arg:
        lines = [f"Голос ветки «{thread_name}»: {current or f'{TTS_SPEAKER} (по умолчанию)'}", ""]
        lines.append("Доступные голоса:")
        for name in TTS_SPEAKERS:
            marker = "✅" if name == (current or TTS_SPEAKER) else "▫️"
            lines.append(f"{marker} {name}")
        lines.append("")
        lines.append("Сменить: /voice <имя>. Сбросить на дефолт из .env: /voice default")
        await message.answer("\n".join(lines))
        return

    if arg in ("default", "сброс", "reset"):
        thread["tts_speaker"] = None
        save_state(state)
        await message.answer(f"Голос ветки «{thread_name}» сброшен на дефолтный ({TTS_SPEAKER}).")
        return

    if arg not in TTS_SPEAKERS:
        await message.answer(
            f"Неизвестный голос «{arg}». Доступные: {', '.join(TTS_SPEAKERS)}. Список — /voice"
        )
        return

    thread["tts_speaker"] = arg
    save_state(state)
    await message.answer(f"Голос ветки «{thread_name}» изменён на «{arg}».")


@dp.message(Command("voicemode"))
async def cmd_voicemode(message: Message, command: CommandObject):
    if not check_access(message):
        return
    arg = (command.args or "").strip().lower()
    state = load_state()
    _, thread_name, thread = get_active_thread(message.chat.id)
    current = thread.get("voice_mode", "assistant")

    if arg in ("transcribe", "on", "off", "только", "расшифровка"):
        new_mode = "transcribe"
    elif arg in ("assistant", "claude", "ассистент"):
        new_mode = "assistant"
    elif not arg:
        # Без аргумента — просто переключаем на противоположный режим.
        new_mode = "transcribe" if current == "assistant" else "assistant"
    else:
        await message.answer(
            "Использование: /voicemode [assistant|transcribe]. Без аргумента — переключает."
        )
        return

    thread["voice_mode"] = new_mode
    save_state(state)
    if new_mode == "transcribe":
        await message.answer(
            f"Голосовые в ветке «{thread_name}»: только расшифровка, Claude не вызывается.\n"
            "Вернуть обычный режим — /voicemode assistant."
        )
    else:
        await message.answer(
            f"Голосовые в ветке «{thread_name}»: обычный режим — расшифровка + ответ Claude голосом.\n"
            "Только расшифровка — /voicemode transcribe."
        )


# ---------- Хендлеры: сообщения ----------

@dp.message(F.text)
async def on_text(message: Message):
    if not check_access(message):
        await message.answer("⛔ У вас нет доступа к этому боту.")
        return
    async with TypingIndicator(message.chat.id):
        reply = await run_claude(message.chat.id, message.text)
    await send_long(message, reply)


def convert_voice_ogg_to_wav(ogg_path: Path, wav_path: Path) -> None:
    """Синхронно конвертирует входящее голосовое .ogg в .wav для Whisper.
    Вызывать только через asyncio.to_thread — subprocess.run() блокирующий и
    не должен выполняться прямо в event loop."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(ogg_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True,
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
    )


@dp.message(F.voice)
async def on_voice(message: Message):
    if not check_access(message):
        return

    ogg_path: Path | None = None
    wav_path: Path | None = None
    try:
        # Узнаём режим ветки заранее, чтобы выбрать правильный статус:
        # если ответ придёт голосом — "записывает голосовое...", иначе,
        # как обычно, "печатает...".
        _, thread_name, thread = get_active_thread(message.chat.id)
        will_reply_with_voice = TTS_ENABLED and thread.get("voice_mode", "assistant") == "assistant"
        indicator_action = ChatAction.RECORD_VOICE if will_reply_with_voice else ChatAction.TYPING

        async with TypingIndicator(message.chat.id, action=indicator_action):
            cwd = thread_dir(message.chat.id, thread_name)
            ogg_path = cwd / f"voice_{uuid.uuid4().hex}.ogg"
            wav_path = ogg_path.with_suffix(".wav")

            log.info("chat=%s: скачиваю голосовое сообщение из Telegram", message.chat.id)
            try:
                file = await asyncio.wait_for(
                    bot.get_file(message.voice.file_id), timeout=VOICE_DOWNLOAD_TIMEOUT
                )
                await asyncio.wait_for(
                    bot.download_file(file.file_path, destination=ogg_path),
                    timeout=VOICE_DOWNLOAD_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.error(
                    "chat=%s: не удалось скачать голосовое сообщение за %sс (таймаут Telegram API)",
                    message.chat.id, VOICE_DOWNLOAD_TIMEOUT,
                )
                await message.answer(
                    "⚠️ Не удалось скачать голосовое сообщение из Telegram (таймаут). Попробуйте ещё раз."
                )
                return

            log.info("chat=%s: конвертирую входящее ogg -> wav через ffmpeg", message.chat.id)
            try:
                # ffmpeg — в отдельном потоке, чтобы не блокировать event loop
                # (иначе "зависает" не только этот чат, но и все остальные,
                # и живой индикатор "печатает..." перестаёт обновляться).
                await asyncio.to_thread(convert_voice_ogg_to_wav, ogg_path, wav_path)
            except subprocess.TimeoutExpired:
                log.error("chat=%s: ffmpeg завис и не завершился за %sс", message.chat.id, FFMPEG_TIMEOUT)
                await message.answer(
                    "⚠️ Конвертация голосового сообщения зависла (таймаут ffmpeg на сервере)."
                )
                return
            except subprocess.CalledProcessError:
                log.exception("chat=%s: ffmpeg не смог конвертировать голосовое сообщение", message.chat.id)
                await message.answer(
                    "⚠️ Не удалось обработать голосовое сообщение (ошибка ffmpeg на сервере). "
                    "Попробуйте отправить его ещё раз или напишите текстом."
                )
                return

            log.info("chat=%s: распознаю текст (Whisper)", message.chat.id)
            try:
                text = await asyncio.to_thread(transcribe, wav_path)
            except Exception:
                log.exception("chat=%s: ошибка распознавания голосового сообщения (whisper)", message.chat.id)
                await message.answer(
                    "⚠️ Не удалось распознать голосовое сообщение (ошибка модели распознавания)."
                )
                return
            log.info("chat=%s: распознано: %r", message.chat.id, text[:200])

            if not text:
                await message.answer("Не удалось распознать голосовое сообщение.")
                return

            await message.answer(f"🎤 Распознано: {text}")

            if thread.get("voice_mode", "assistant") == "transcribe":
                # /voicemode transcribe — только расшифровка, Claude не
                # вызывается и ответом голосом заниматься не нужно.
                return

            reply = await run_claude(message.chat.id, text)
            # Голосовой вход -> голосовой ответ (зеркально). Синтез сам по
            # себе надёжно откатывается на текст при ошибке внутри send_reply.
            # Остаёмся внутри TypingIndicator — синтез может занять время.
            await send_reply(message, reply, cwd, as_voice=True, speaker=thread.get("tts_speaker"))
    except Exception:
        # Раньше любое необработанное исключение (например, сбой скачивания
        # файла из Telegram) тихо гасилось где-то в диспетчере: пользователь
        # видел "печатает..." и затем — ничего. Теперь бот всегда отвечает.
        log.exception("chat=%s: непредвиденная ошибка при обработке голосового сообщения", message.chat.id)
        await message.answer("⚠️ Произошла ошибка при обработке голосового сообщения. Попробуйте ещё раз.")
    finally:
        # Раньше временные файлы удалялись только на успешном пути — при любой
        # ошибке выше .ogg/.wav оставались в рабочей директории навсегда.
        if ogg_path is not None:
            ogg_path.unlink(missing_ok=True)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


@dp.message(F.photo)
async def on_photo(message: Message):
    if not check_access(message):
        return
    async with TypingIndicator(message.chat.id):
        _, thread_name, _ = get_active_thread(message.chat.id)
        cwd = thread_dir(message.chat.id, thread_name)
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

