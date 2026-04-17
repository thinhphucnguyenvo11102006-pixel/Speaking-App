import streamlit as st
import google.generativeai as genai
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
from io import BytesIO
import base64
import re

# ─────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────
GEMINI_MODEL  = "gemini-2.5-flash"
WHISPER_MODEL = "whisper-large-v3"
PART1_Q_LIMIT = 4   # questions before advancing to Part 2
PART3_Q_LIMIT = 4   # questions in Part 3 before ending the exam

# ─────────────────────────────────────────────────────────
# 2. PAGE CONFIG  (must be the FIRST Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IELTS Speaking Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# 3. API CLIENTS  (cached so they are created only once)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def init_clients():
    """Initialise Gemini + Groq clients using Streamlit Secrets."""
    try:
        g_key = st.secrets["GOOGLE_API_KEY"]
        q_key = st.secrets["GROQ_API_KEY"]
    except KeyError as e:
        st.error(
            f"❌ Missing secret: **{e}**. "
            "Add your API keys in **Streamlit Cloud → App Settings → Secrets**, "
            "or create `.streamlit/secrets.toml` for local development."
        )
        st.stop()
    genai.configure(api_key=g_key)
    return genai.GenerativeModel(GEMINI_MODEL), Groq(api_key=q_key)

model, groq_client = init_clients()

# ─────────────────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────────────────
DEFAULTS: dict = {
    "chat_history":      [],
    "mic_key":           0,
    "last_audio_memory": None,
    "audio_to_play":     None,
    "last_tts_text":     None,
    "part":              1,      # 1 | 2 | 3
    "q_count":           0,      # answers submitted in current part
    "cue_card":          None,   # Part-2 cue card text
    "scores":            [],     # list[float] – one per evaluated turn
    "exam_ended":        False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
# 5. PROMPTS
# ─────────────────────────────────────────────────────────
BASE_SYSTEM = """
You are a professional IELTS Speaking Examiner conducting an official speaking test.

OFFICIAL IELTS SCORING RUBRIC (score each criterion out of 9):
- Fluency & Coherence (FC): smoothness, hesitation, self-correction, discourse markers.
- Lexical Resource (LR): range, accuracy, and appropriacy of vocabulary.
- Grammatical Range & Accuracy (GRA): variety of structures, error frequency.
- Pronunciation (P): clarity, stress patterns, intonation, intelligibility.
  ⚠ If the input contains "[PRONUNCIATION ERROR: X → Y]", the candidate mispronounced X (meaning Y). Penalise P accordingly and continue the conversation using meaning Y.

MANDATORY RESPONSE FORMAT — TWO BLOCKS SEPARATED BY |||:

Block 1 — FEEDBACK (shown as text, never spoken):
**Overall Band: [X.X / 9]**
| Criterion | Band | Comment |
|---|---|---|
| Fluency & Coherence | [score] | [concise remark] |
| Lexical Resource | [score] | [concise remark] |
| Grammatical Range | [score] | [concise remark] |
| Pronunciation | [score] | [concise remark] |
💡 **Tip:** [one clear, actionable improvement tip]

|||

Block 2 — NEXT TURN (spoken aloud — friendly, encouraging, 1-3 sentences):
[Natural follow-up question or transition phrase]

ALWAYS include the ||| separator, even if feedback is brief.
"""

PART_CONTEXTS = {
    1: "EXAM PART: Part 1 — Familiar topics (home, hobbies, work, family). Keep questions short, friendly, and conversational.",
    2: "EXAM PART: Part 2 — Long turn. The candidate has just delivered their 1–2 minute talk on the cue card. Evaluate the extended response then smoothly transition to Part 3.",
    3: "EXAM PART: Part 3 — Abstract discussion. Ask deeper, thought-provoking questions linked to the Part 2 topic. Encourage complex, opinion-based answers.",
}

START_PROMPT = (
    BASE_SYSTEM + "\n" + PART_CONTEXTS[1] + """

TASK: Begin the exam. Welcome the candidate warmly, briefly explain the format, then ask the first Part 1 question on a familiar topic.
Do NOT provide a feedback table for this opening message — just the greeting and first question (no ||| separator needed).
"""
)

CUE_CARD_PROMPT = """
Generate one IELTS Part 2 cue card.

Return ONLY the cue card, formatted exactly like this:
---
**Topic:** [interesting topic]

Describe [something specific and interesting].

You should say:
• [point 1]
• [point 2]
• [point 3]
• and explain [point 4 — opinion/reflection]
---
"""

# ─────────────────────────────────────────────────────────
# 6. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def text_to_speech(text: str) -> None:
    """Convert text to speech and autoplay silently in the browser."""
    try:
        tts = gTTS(text=text, lang="en", tld="co.uk")
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(
            f'<audio autoplay="true" style="display:none;">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )
        st.session_state.last_tts_text = text
    except Exception as e:
        st.warning(f"TTS error — {e}")


def whisper_stt(audio_bytes: bytes) -> str | None:
    """Transcribe audio bytes using Groq Whisper."""
    try:
        return groq_client.audio.transcriptions.create(
            file=("input.wav", audio_bytes),
            model=WHISPER_MODEL,
            response_format="text",
            language="en",
        )
    except Exception as e:
        st.error(f"Transcription error — {e}")
        return None


def repair_transcription(raw: str) -> str:
    """Flag phonetic mis-transcriptions without changing grammar."""
    prompt = f"""
Act as a Speech-to-Text error checker.

Raw Input: "{raw}"

Rules:
1. Identify phonetic mis-transcription errors only (e.g., "bitch" heard instead of "beach").
2. If error found: correct it AND append "[PRONUNCIATION ERROR: <wrong> → <right>]".
3. If no error: return the original text exactly, unchanged.
4. CRITICAL: Do NOT fix grammar, word choice, or style. Only fix mishear errors.
"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return raw


def parse_band_score(text: str) -> float | None:
    """Extract the overall band score from an AI feedback block."""
    m = re.search(r"\*\*Overall Band:\s*([\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def advance_part() -> None:
    """Increment part counter and reset question count."""
    st.session_state.part += 1
    st.session_state.q_count = 0


def reset_exam() -> None:
    """Wipe all session state back to defaults."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def build_prompt(user_answer: str) -> str:
    """Assemble the full examiner prompt for the current part."""
    ctx = PART_CONTEXTS.get(st.session_state.part, PART_CONTEXTS[3])
    return BASE_SYSTEM + "\n" + ctx + f"\n\nCandidate Answer: {user_answer}"


def speak(text: str) -> None:
    """Schedule text for TTS playback after the next rerun."""
    st.session_state.audio_to_play = text


def process_answer(user_content: str) -> None:
    """Full pipeline: transcription repair → history → AI evaluation → TTS."""
    if not user_content.strip():
        return

    # Step 1: repair & flag pronunciation errors
    with st.spinner("Analysing your answer…"):
        repaired = repair_transcription(user_content)

    if "[PRONUNCIATION ERROR" in repaired:
        display_text = repaired.split("[PRONUNCIATION ERROR")[0].strip()
        error_note   = repaired.split("[PRONUNCIATION ERROR")[1].rstrip("]")
        st.toast(f"⚠️ Pronunciation note:{error_note}", icon="🗣️")
    else:
        display_text = repaired

    st.session_state.chat_history.append({"role": "user", "content": display_text})

    # Step 2: get AI examiner response
    with st.spinner("Examiner is evaluating…"):
        try:
            response = model.generate_content(build_prompt(repaired))
            reply    = response.text

            if "|||" in reply:
                feedback_raw, question_raw = reply.split("|||", 1)
                feedback = feedback_raw.strip()
                question = question_raw.strip()
            else:
                feedback = ""
                question = reply.strip()

            # Store feedback + update score
            if feedback:
                score = parse_band_score(feedback)
                if score is not None:
                    st.session_state.scores.append(score)
                st.session_state.chat_history.append({"role": "feedback", "content": feedback})

            # Step 3: part logic
            st.session_state.q_count += 1

            if st.session_state.part == 1 and st.session_state.q_count >= PART1_Q_LIMIT:
                # Generate cue card and transition to Part 2
                with st.spinner("Preparing your Part 2 cue card…"):
                    cue = model.generate_content(CUE_CARD_PROMPT).text.strip()
                st.session_state.cue_card = cue
                transition = (
                    "Excellent work on Part 1! Now we'll move to Part 2 — the long turn. "
                    "Please read your cue card carefully. You have about one minute to prepare, "
                    "then speak for one to two minutes."
                )
                st.session_state.chat_history.append({"role": "part_transition", "content": "📋 Part 2 — Long Turn"})
                st.session_state.chat_history.append({"role": "assistant", "content": transition})
                speak(transition)
                advance_part()

            elif st.session_state.part == 2:
                # After the Part 2 monologue, move to Part 3
                transition = (
                    "Thank you — that was a great response! "
                    "We'll now move on to Part 3, where I'll ask some broader questions on a related theme."
                )
                st.session_state.chat_history.append({"role": "part_transition", "content": "💬 Part 3 — Discussion"})
                st.session_state.chat_history.append({"role": "assistant", "content": transition})
                speak(transition)
                advance_part()

            elif st.session_state.part == 3 and st.session_state.q_count >= PART3_Q_LIMIT:
                # End of exam
                ending = (
                    "And that brings us to the end of the IELTS Speaking test. "
                    "Well done for completing all three parts! "
                    "Please check your score summary in the sidebar on the left."
                )
                st.session_state.chat_history.append({"role": "assistant", "content": ending})
                speak(ending)
                st.session_state.exam_ended = True

            else:
                # Normal next question
                if question:
                    st.session_state.chat_history.append({"role": "assistant", "content": question})
                    speak(question)

        except Exception as e:
            st.error(f"AI error — {e}")

# ─────────────────────────────────────────────────────────
# 7. FIRST-LOAD INITIALISATION
# ─────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    try:
        resp    = model.generate_content(START_PROMPT)
        opening = resp.text.replace("|||", "").strip()
        st.session_state.chat_history.append({"role": "assistant", "content": opening})
        speak(opening)
    except Exception as e:
        st.error(f"Initialisation error — {e}")

# ─────────────────────────────────────────────────────────
# 8. SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 IELTS Speaking Pro")
    st.caption("AI-powered exam simulator")
    st.divider()

    # Part progress
    st.markdown("**📍 Exam Progress**")
    part_labels = {
        1: "Part 1 — Familiar Topics",
        2: "Part 2 — Long Turn",
        3: "Part 3 — Discussion",
    }
    cur = st.session_state.part
    for p, label in part_labels.items():
        if p < cur:
            icon = "✅"
        elif p == cur:
            icon = "🟢"
        else:
            icon = "⬜"
        st.markdown(f"{icon} {label}")

    st.divider()

    # Running score
    scores = st.session_state.scores
    if scores:
        avg = sum(scores) / len(scores)
        st.metric("⭐ Average Band Score", f"{avg:.1f} / 9.0")
        st.caption(f"Based on {len(scores)} evaluated response(s)")
    else:
        st.info("Your scores will appear here after your first answer.")

    st.divider()

    # Controls
    if st.session_state.last_tts_text:
        if st.button("🔊 Replay Last Question", use_container_width=True):
            text_to_speech(st.session_state.last_tts_text)

    if st.button("🔄 Reset Exam", type="secondary", use_container_width=True):
        reset_exam()
        st.rerun()

# ─────────────────────────────────────────────────────────
# 9. MAIN CHAT AREA
# ─────────────────────────────────────────────────────────
st.header("IELTS Speaking Simulator", divider="gray")

# Part 2 cue card (pinned below header when available)
if st.session_state.part >= 2 and st.session_state.cue_card:
    with st.expander("📋 Part 2 Cue Card", expanded=(st.session_state.part == 2)):
        st.markdown(st.session_state.cue_card)

# Render chat history
for msg in st.session_state.chat_history:
    role    = msg["role"]
    content = msg["content"]

    if role == "user":
        with st.chat_message("user"):
            st.write(content)

    elif role == "assistant":
        with st.chat_message("assistant", avatar="🎓"):
            st.write(content)

    elif role == "feedback":
        with st.expander("📊 Examiner Feedback & Score", expanded=True):
            st.markdown(content)

    elif role == "part_transition":
        st.info(f"**{content}**", icon="📌")

# ─────────────────────────────────────────────────────────
# 10. AUTOPLAY TTS
# ─────────────────────────────────────────────────────────
if st.session_state.audio_to_play:
    text_to_speech(st.session_state.audio_to_play)
    st.session_state.audio_to_play = None

# ─────────────────────────────────────────────────────────
# 11. INPUT AREA
# ─────────────────────────────────────────────────────────
if not st.session_state.exam_ended:
    st.write("---")
    tab_voice, tab_text = st.tabs(["🎙️ Voice Recording", "⌨️ Type Answer"])

    with tab_voice:
        st.caption("Press **Start Recording**, speak your answer, then press **Stop Recording**.")
        audio_data = mic_recorder(
            start_prompt="▶ Start Recording",
            stop_prompt="⏹ Stop Recording",
            key=str(st.session_state.mic_key),
            format="wav",
        )
        if audio_data and "bytes" in audio_data:
            st.session_state.last_audio_memory = audio_data["bytes"]
            raw = whisper_stt(audio_data["bytes"])
            if raw:
                with st.expander("📝 Transcription (what we heard)", expanded=True):
                    st.write(raw)
                process_answer(raw)
                st.session_state.mic_key += 1
                st.rerun()

    with tab_text:
        with st.form("text_input_form", clear_on_submit=True):
            user_text = st.text_area(
                "Type your answer here:",
                placeholder="e.g. I really enjoy spending time outdoors because…",
                height=130,
            )
            submitted = st.form_submit_button(
                "✅ Submit Answer", type="primary", use_container_width=True
            )
        if submitted and user_text:
            process_answer(user_text)
            st.rerun()

else:
    # Exam complete screen
    st.success("🎉 **Exam Complete!** Check your score summary in the sidebar.", icon="🏆")
    if st.button("🔄 Start a New Exam", type="primary"):
        reset_exam()
        st.rerun()