# tts_service.py (updated)
import asyncio
import logging
import time
import tempfile
import os
import os.path
from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
import xml.sax.saxutils as saxutils
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("✗ langdetect not available - install: pip install langdetect")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auto-detect available TTS backends
TTS_BACKENDS = []

try:
    from gtts import gTTS
    import pygame
    TTS_BACKENDS.append("gtts")
    logger.info("✓ gTTS backend available")
except ImportError:
    logger.warning("✗ gTTS or pygame not available - install: pip install gtts pygame")

try:
    import edge_tts
    TTS_BACKENDS.append("edge_tts")
    logger.info("✓ Edge TTS backend available")
except ImportError:
    logger.warning("✗ Edge TTS not available - install: pip install edge-tts")

# Check for ffmpeg
try:
    import subprocess
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    logger.info("✓ ffmpeg available")
except (subprocess.CalledProcessError, FileNotFoundError):
    logger.warning("✗ ffmpeg not available - install ffmpeg for reliable audio processing")

class IndianLanguage(Enum):
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    URDU = "ur"
    ODIA = "or"
    ASSAMESE = "as"
    MAITHILI = "mai"
    SANTALI = "sat"
    KASHMIRI = "ks"
    NEPALI = "ne"
    SANSKRIT = "sa"
    SINDHI = "sd"
    KONKANI = "kok"
    MANIPURI = "mni"
    DOGRI = "doi"
    BODO = "brx"
    ENGLISH = "en"

class VoiceGender(Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass
class TTSConfig:
    backend: str = "auto"
    voice_gender: VoiceGender = VoiceGender.FEMALE
    voice: str | None = None
    speech_rate: str = "medium"
    volume: float = 0.9
    timeout: float = 30.0

@dataclass
class LatencyMetrics:
    text_processing: float = 0.0
    network_latency: float = 0.0
    synthesis_time: float = 0.0
    audio_playback: float = 0.0
    total_time: float = 0.0
    text_length: int = 0
    characters_per_sec: float = 0.0

    def __str__(self):
        return (f"\n⏱️  LATENCY REPORT:\n"
                f"  Text Length: {self.text_length} chars\n"
                f"  Text Processing: {self.text_processing:.3f}s\n"
                f"  Network Latency: {self.network_latency:.3f}s\n"
                f"  Synthesis Time: {self.synthesis_time:.3f}s\n"
                f"  Audio Playback: {self.audio_playback:.3f}s\n"
                f"  TOTAL TIME: {self.total_time:.3f}s\n"
                f"  Speed: {self.characters_per_sec:.1f} chars/sec")

class IndianTextPreprocessor:
    def __init__(self):
        self.abbreviations = {
            'डॉ.': 'डॉक्टर', 'श्री.': 'श्रीमान', 'श्रीमती.': 'श्रीमती',
            'एस.': 'एस', 'पी.': 'पी', 'एम.': 'एम', 'के.': 'के',
            'आदि.': 'आदि', 'इत्यादि.': 'इत्यादि', 'उर्फ़.': 'उर्फ़',
            'अं.': 'अंक', 'पृ.': 'पृष्ठ', 'प्रो.': 'प्रोफेसर'
        }
        self.number_map = {
            '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
            '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
            '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह',
            '14': 'चौदह', '15': 'पंद्रह', '16': 'सोलह', '17': 'सत्रह',
            '18': 'अठारह', '19': 'उन्नीस', '20': 'बीस', '21': 'इक्कीस',
            '22': 'बाईस', '23': 'तेईस', '24': 'चौबीस', '25': 'पच्चीस',
            '26': 'छब्बीस', '27': 'सत्ताईस', '28': 'अट्ठाईस', '29': 'उनतीस',
            '30': 'तीस', '31': 'इकतीस', '32': 'बत्तीस', '33': 'तैंतीस',
            '34': 'चौंतीस', '35': 'पैंतीस', '36': 'छत्तीस', '37': 'सैंतीस',
            '38': 'अड़तीस', '39': 'उनतालीस', '40': 'चालीस', '41': 'इकतालीस',
            '42': 'बयालीस', '43': 'तैंतालीस', '44': 'चवालीस', '45': 'पैंतालीस',
            '46': 'छयालिस', '47': 'सैंतालीस', '48': 'अड़तालीस', '49': 'उनचास',
            '50': 'पचास', '51': 'इक्यावन', '52': 'बावन', '53': 'तिरेपन',
            '54': 'चौवन', '55': 'पचपन', '56': 'छप्पन', '57': 'सत्तावन',
            '58': 'अट्ठावन', '59': 'उनसठ', '60': 'साठ', '61': 'इकसठ',
            '62': 'बासठ', '63': 'तिरेसठ', '64': 'चौंसठ', '65': 'पैंसठ',
            '66': 'छियासठ', '67': 'सड़सठ', '68': 'अड़सठ', '69': 'उनहत्तर',
            '70': 'सत्तर', '71': 'इकहत्तर', '72': 'बहत्तर', '73': 'तिहत्तर',
            '74': 'चौहत्तर', '75': 'पचहत्तर', '76': 'छिहत्तर', '77': 'सतहत्तर',
            '78': 'अठहत्तर', '79': 'उन्यासी', '80': 'अस्सी', '81': 'इक्यासी',
            '82': 'बयासी', '83': 'तिरासी', '84': 'चौरासी', '85': 'पचासी',
            '86': 'छियासी', '87': 'सत्तासी', '88': 'अठासी', '89': 'नवासी',
            '90': 'नब्बे', '91': 'इक्यानबे', '92': 'बानवे', '93': 'तिरानबे',
            '94': 'चौरानबे', '95': 'पंचानबे', '96': 'छियानबे', '97': 'सत्तानबे',
            '98': 'अट्ठानबे', '99': 'निन्यानबे', '100': 'सौ'
        }

    def clean_text(self, text: str, language: IndianLanguage = IndianLanguage.HINDI) -> Tuple[str, float]:
        start_time = time.time()
        if not text.strip():
            return "", time.time() - start_time

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        if language == IndianLanguage.HINDI:
            for abbr, expansion in self.abbreviations.items():
                text = text.replace(abbr, expansion)
            text = re.sub(r'\b\d+\b', lambda m: self._indian_number_to_words(m.group()), text)

        text = re.sub(r'[.!?]', r'\g<0> ', text)
        text = re.sub(r'[,;:]', r'\g<0> ', text)
        text = saxutils.escape(text)

        processing_time = time.time() - start_time
        return text.strip(), processing_time

    def _indian_number_to_words(self, number_str: str) -> str:
        try:
            num = int(number_str)
            if 0 <= num <= 20:
                return self.number_map.get(str(num), number_str)
            elif 21 <= num <= 99:
                tens = num // 10
                units = num % 10
                if units == 0:
                    return f"{self.number_map.get(str(tens*10), str(tens*10))}"
                else:
                    return f"{self.number_map.get(str(tens*10), str(tens*10))} {self.number_map.get(str(units), str(units))}"
            else:
                return number_str
        except:
            return number_str

    def detect_language(self, text: str) -> str:
        if not LANGDETECT_AVAILABLE:
            logger.warning("Language detection not available, defaulting to Hindi")
            return "hi"
        try:
            lang = detect(text)
            lang_map = {
                'hi': 'hi', 'bn': 'bn', 'ta': 'ta', 'te': 'te', 'mr': 'mr',
                'gu': 'gu', 'kn': 'kn', 'ml': 'ml', 'pa': 'pa', 'ur': 'ur',
                'or': 'or', 'as': 'as', 'ne': 'ne', 'en': 'en'
            }
            return lang_map.get(lang, 'hi')
        except:
            logger.warning("Language detection failed, defaulting to Hindi")
            return 'hi'

class IndianTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.backend = self._select_backend()
        self.preprocessor = IndianTextPreprocessor()
        self.pygame_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_engine()
        logger.info(f"Initialized IndianTTS with {self.backend} backend")

    def _select_backend(self) -> str:
        if self.config.backend != "auto":
            if self.config.backend in TTS_BACKENDS:
                return self.config.backend
            logger.warning(f"Backend {self.config.backend} not available, auto-selecting...")
        if "edge_tts" in TTS_BACKENDS:
            return "edge_tts"
        return "gtts" if "gtts" in TTS_BACKENDS else ""

    def _initialize_pygame(self):
        if not self.pygame_initialized:
            try:
                import pygame
                pygame.mixer.quit()
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
                pygame.mixer.init()
                self.pygame_initialized = True
                logger.info("Pygame mixer initialized for low latency")
            except Exception as e:
                logger.error(f"Pygame init failed: {e}")
                raise RuntimeError("Failed to initialize pygame for audio playback")

    def _initialize_engine(self):
        if not self.backend:
            raise RuntimeError("No TTS backend available")
        if self.backend in ["gtts", "edge_tts"]:
            try:
                import pygame
                self._initialize_pygame()
            except ImportError:
                raise RuntimeError("pygame is required for audio playback")

    async def validate_voice(self, voice: str, language: IndianLanguage) -> str:
        if not voice or self.backend != "edge_tts":
            return self._get_default_voice(language)
        try:
            voices = await list_available_voices()
            if voice in [v['Name'] for v in voices]:
                return voice
            logger.warning(f"Invalid voice {voice} for {language.name}, using default")
            return self._get_default_voice(language)
        except Exception as e:
            logger.error(f"Voice validation failed: {e}, using default")
            return self._get_default_voice(language)

    def _get_default_voice(self, language: IndianLanguage) -> str:
        voice_map = {
            IndianLanguage.HINDI: "hi-IN-SwaraNeural" if self.config.voice_gender == VoiceGender.FEMALE else "hi-IN-MadhurNeural",
            IndianLanguage.BENGALI: "bn-IN-TanishaaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "bn-IN-BashkarNeural",
            IndianLanguage.TAMIL: "ta-IN-PallaviNeural" if self.config.voice_gender == VoiceGender.FEMALE else "ta-IN-ValluvarNeural",
            IndianLanguage.TELUGU: "te-IN-ShrutiNeural" if self.config.voice_gender == VoiceGender.FEMALE else "te-IN-MohanNeural",
            IndianLanguage.MARATHI: "mr-IN-AarohiNeural" if self.config.voice_gender == VoiceGender.FEMALE else "mr-IN-ManoharNeural",
            IndianLanguage.GUJARATI: "gu-IN-DhwaniNeural" if self.config.voice_gender == VoiceGender.FEMALE else "gu-IN-NiranjanNeural",
            IndianLanguage.KANNADA: "kn-IN-SapnaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "kn-IN-GaganNeural",
            IndianLanguage.MALAYALAM: "ml-IN-SobhanaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "ml-IN-MidhunNeural",
            IndianLanguage.PUNJABI: "pa-IN-KalpanaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "pa-IN-HarmanNeural",
            IndianLanguage.URDU: "ur-IN-RiyaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "ur-IN-SalmanNeural",
            IndianLanguage.ODIA: "or-IN-SayakNeural" if self.config.voice_gender == VoiceGender.FEMALE else "or-IN-SambitNeural",
            IndianLanguage.ASSAMESE: "as-IN-RupaliNeural" if self.config.voice_gender == VoiceGender.FEMALE else "as-IN-GunakarNeural",
            IndianLanguage.NEPALI: "ne-NP-AnjuNeural" if self.config.voice_gender == VoiceGender.FEMALE else "ne-NP-BipinNeural",
            IndianLanguage.ENGLISH: "en-IN-NeerjaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "en-IN-PrabhatNeural",
            IndianLanguage.MAITHILI: "hi-IN-SwaraNeural",
            IndianLanguage.SANTALI: "hi-IN-SwaraNeural",
            IndianLanguage.KASHMIRI: "hi-IN-SwaraNeural",
            IndianLanguage.SANSKRIT: "hi-IN-SwaraNeural",
            IndianLanguage.SINDHI: "gu-IN-DhwaniNeural",
            IndianLanguage.KONKANI: "mr-IN-AarohiNeural",
            IndianLanguage.MANIPURI: "as-IN-RupaliNeural",
            IndianLanguage.DOGRI: "pa-IN-KalpanaNeural",
            IndianLanguage.BODO: "as-IN-RupaliNeural"
        }
        return voice_map.get(language, "hi-IN-SwaraNeural")

    async def _speak_async(self, text: str, language: IndianLanguage, voice: str | None = None) -> Tuple[bool, LatencyMetrics]:
        metrics = LatencyMetrics()
        total_start = time.time()

        if not text.strip():
            logger.warning("Received empty text input")
            return False, metrics

        clean_text, text_processing_time = self.preprocessor.clean_text(text, language)
        metrics.text_processing = text_processing_time
        metrics.text_length = len(clean_text)

        if not clean_text:
            logger.warning("Text preprocessing resulted in empty output")
            return False, metrics

        try:
            synthesis_start = time.time()
            network_start = time.time()

            if self.backend == "gtts":
                success, playback_time = self._speak_gtts(clean_text, language)
            elif self.backend == "edge_tts":
                validated_voice = await self.validate_voice(voice, language)
                success, playback_time = await self._speak_edge_tts(clean_text, language, validated_voice)
            else:
                logger.error("No valid TTS backend available")
                return False, metrics

            metrics.network_latency = time.time() - network_start
            metrics.synthesis_time = time.time() - synthesis_start - playback_time
            metrics.audio_playback = playback_time
            metrics.total_time = time.time() - total_start
            metrics.characters_per_sec = metrics.text_length / metrics.total_time if metrics.total_time > 0 else 0.0

            return success, metrics

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            metrics.total_time = time.time() - total_start
            return False, metrics

    async def speak(self, text: str, language: IndianLanguage = IndianLanguage.HINDI, voice: str | None = None) -> Tuple[bool, LatencyMetrics]:
        return await self._speak_async(text, language, voice)

    def _speak_gtts(self, text: str, language: IndianLanguage) -> Tuple[bool, float]:
        try:
            from gtts import gTTS
            import pygame

            if not self.pygame_initialized:
                self._initialize_pygame()

            tts_start = time.time()
            tts = gTTS(text=text, lang=language.value, slow=False)

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
                tts.save(temp_path)

                try:
                    playback_start = time.time()
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(100)
                    playback_time = time.time() - playback_start
                    return True, playback_time
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logger.warning(f"Could not delete temp file: {e}")

        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return False, 0.0

    async def _speak_edge_tts(self, text: str, language: IndianLanguage, voice: str) -> Tuple[bool, float]:
        try:
            import edge_tts
            import pygame

            if not self.pygame_initialized:
                self._initialize_pygame()

            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                logger.error(f"No audio data received from Edge TTS for voice {voice} and language {language.name}")
                return False, 0.0

            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"indiantts_{os.getpid()}_{int(time.time())}.mp3")

            try:
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                if not os.path.exists(temp_path):
                    raise IOError("Temp file creation failed")

                playback_start = time.time()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(100)
                playback_time = time.time() - playback_start
                return True, playback_time
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return False, 0.0

    async def _save_edge_tts(self, text: str, output_path: str, language: IndianLanguage, voice: str | None = None) -> bool:
        try:
            import edge_tts
            import os

            validated_voice = await self.validate_voice(voice, language)
            communicate = edge_tts.Communicate(text, validated_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                logger.error(f"No audio data received from Edge TTS for voice {validated_voice} and language {language.name}")
                return False

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            return True

        except Exception as e:
            logger.error(f"Edge TTS save error: {e}")
            return False

    def save_to_file(self, text: str, output_path: str, language: IndianLanguage = IndianLanguage.HINDI, voice: str | None = None) -> Tuple[bool, LatencyMetrics]:
        metrics = LatencyMetrics()
        total_start = time.time()

        clean_text, text_processing_time = self.preprocessor.clean_text(text, language)
        metrics.text_processing = text_processing_time
        metrics.text_length = len(clean_text)

        if not clean_text:
            logger.warning("Text preprocessing resulted in empty output")
            return False, metrics

        try:
            synthesis_start = time.time()

            if self.backend == "gtts":
                success = self._save_gtts(clean_text, output_path, language)
            elif self.backend == "edge_tts":
                success = asyncio.run(self._save_edge_tts(clean_text, output_path, language, voice))
            else:
                logger.warning(f"Save not supported for {self.backend}")
                return False, metrics

            metrics.synthesis_time = time.time() - synthesis_start
            metrics.total_time = time.time() - total_start
            metrics.characters_per_sec = metrics.text_length / metrics.total_time if metrics.total_time > 0 else 0.0

            return success, metrics

        except Exception as e:
            logger.error(f"Save failed: {e}")
            metrics.total_time = time.time() - total_start
            return False, metrics

    def _save_gtts(self, text: str, output_path: str, language: IndianLanguage) -> bool:
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=language.value, slow=False)
            tts.save(output_path)
            return True
        except Exception as e:
            logger.error(f"gTTS save error: {e}")
            return False

async def list_available_voices():
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        indian_voices = [
            {
                'Name': v['Name'],
                'Gender': v.get('Gender', 'Unknown'),
                'Language': v.get('Language', 'Unknown'),
                'Description': {
                    'hi-IN-SwaraNeural': 'Female, North Indian, formal',
                    'hi-IN-MadhurNeural': 'Male, North Indian, conversational',
                    'ta-IN-PallaviNeural': 'Female, South Indian, melodic',
                    'ta-IN-ValluvarNeural': 'Male, South Indian, expressive',
                    'bn-IN-TanishaaNeural': 'Female, East Indian, melodic',
                    'bn-IN-BashkarNeural': 'Male, East Indian, clear',
                    'te-IN-ShrutiNeural': 'Female, South Indian, smooth',
                    'te-IN-MohanNeural': 'Male, South Indian, dynamic',
                    'mr-IN-AarohiNeural': 'Female, West Indian, crisp',
                    'mr-IN-ManoharNeural': 'Male, West Indian, steady',
                    'gu-IN-DhwaniNeural': 'Female, West Indian, vibrant',
                    'gu-IN-NiranjanNeural': 'Male, West Indian, formal',
                    'kn-IN-SapnaNeural': 'Female, South Indian, flowing',
                    'kn-IN-GaganNeural': 'Male, South Indian, resonant',
                    'ml-IN-SobhanaNeural': 'Female, South Indian, soft',
                    'ml-IN-MidhunNeural': 'Male, South Indian, precise',
                    'pa-IN-KalpanaNeural': 'Female, Northwest Indian, energetic',
                    'pa-IN-HarmanNeural': 'Male, Northwest Indian, bold',
                    'ur-IN-RiyaNeural': 'Female, Indo-Pak, poetic',
                    'ur-IN-SalmanNeural': 'Male, Indo-Pak, narrative',
                    'or-IN-SayakNeural': 'Female, East Indian, gentle',
                    'or-IN-SambitNeural': 'Male, East Indian, firm',
                    'as-IN-RupaliNeural': 'Female, Northeast Indian, lilting',
                    'as-IN-GunakarNeural': 'Male, Northeast Indian, steady',
                    'ne-NP-AnjuNeural': 'Female, Himalayan, clear',
                    'ne-NP-BipinNeural': 'Male, Himalayan, warm',
                    'en-IN-NeerjaNeural': 'Female, Indian English, clear',
                    'en-IN-PrabhatNeural': 'Male, Indian English, professional'
                }.get(v['Name'], 'Standard accent')
            }
            for v in voices
            if any(lang in v['Name'] for lang in ['hi-IN', 'bn-IN', 'ta-IN', 'te-IN', 'mr-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'pa-IN', 'ur-IN', 'or-IN', 'as-IN', 'ne-NP', 'en-IN'])
        ]
        return indian_voices
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        return []

# FastAPI app
app = FastAPI(title="Indian TTS API")

# Add CORS middleware (corrected)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"  # Optional for testing; remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class TTSRequest(BaseModel):
    text: str
    language: str = "hi"
    voice_gender: str = "female"
    backend: str = "auto"
    voice: Optional[str] = None

# Initialize TTS engine
tts_config = TTSConfig(
    backend="auto",
    voice_gender=VoiceGender.FEMALE,
    speech_rate="medium",
    volume=0.9
)
tts_engine = IndianTTS(tts_config)

@app.post("/tts/speak")
async def speak(request: TTSRequest):
    try:
        language = next(
            (lang for lang in IndianLanguage if lang.value == request.language or lang.name.lower() == request.language.lower()),
            IndianLanguage.HINDI
        )
        voice_gender = VoiceGender(request.voice_gender.lower()) if request.voice_gender.lower() in [e.value for e in VoiceGender] else VoiceGender.FEMALE
        tts_engine.config.voice_gender = voice_gender
        tts_engine.config.backend = request.backend
        tts_engine.config.voice = request.voice
        success, metrics = await tts_engine.speak(request.text, language, request.voice)
        if not success:
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        return {
            "success": success,
            "metrics": {
                "text_processing": metrics.text_processing,
                "network_latency": metrics.network_latency,
                "synthesis_time": metrics.synthesis_time,
                "audio_playback": metrics.audio_playback,
                "total_time": metrics.total_time,
                "text_length": metrics.text_length,
                "characters_per_sec": metrics.characters_per_sec
            }
        }
    except Exception as e:
        logger.error(f"Error in /tts/speak: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/save")
async def save(request: TTSRequest):
    try:
        language = next(
            (lang for lang in IndianLanguage if lang.value == request.language or lang.name.lower() == request.language.lower()),
            IndianLanguage.HINDI
        )
        voice_gender = VoiceGender(request.voice_gender.lower()) if request.voice_gender.lower() in [e.value for e in VoiceGender] else VoiceGender.FEMALE
        tts_engine.config.voice_gender = voice_gender
        tts_engine.config.backend = request.backend
        tts_engine.config.voice = request.voice
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            output_path = tmp_file.name
        success, metrics = tts_engine.save_to_file(request.text, output_path, language, request.voice)
        if not success:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename="tts_output.mp3",
            headers={"X-Latency-Metrics": str(metrics)}
        )
    except Exception as e:
        logger.error(f"Error in /tts/save: {e}")
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/voices")
async def list_voices():
    try:
        if tts_engine.backend == "edge_tts":
            voices = await list_available_voices()
            return {"voices": voices}
        else:
            return {"message": "Voice list only available for edge_tts backend"}
    except Exception as e:
        logger.error(f"Error in /tts/voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def interactive_demo():
    print("\n🎤 भारतीय भाषा पाठ-से-वाणी प्रणाली | Indian Language Text-to-Speech System")
    print("=" * 70)

    if not TTS_BACKENDS:
        print("❌ कोई TTS बैकेंड उपलब्ध नहीं है! | No TTS backends available!")
        print("कृपया इंस्टॉल करें: pip install edge-tts gtts pygame langdetect")
        print("ffmpeg की भी आवश्यकता हो सकती है | ffmpeg may also be required")
        return

    print(f"उपलब्ध बैकेंड: {', '.join(TTS_BACKENDS)}")

    config = TTSConfig(
        backend="auto",
        voice_gender=VoiceGender.FEMALE,
        speech_rate="medium",
        volume=0.9
    )

    try:
        tts = IndianTTS(config)
        preprocessor = IndianTextPreprocessor()
        print(f"✓ TTS इंजन प्रारंभ किया गया | TTS engine initialized ({tts.backend})")
    except Exception as e:
        print(f"❌ TTS इंजन प्रारंभ करने में विफल | Failed to initialize TTS: {e}")
        print("कृपया सुनिश्चित करें कि pygame और ffmpeg इंस्टॉल हैं | Ensure pygame and ffmpeg are installed")
        return

    print("\n📝 उपयोग निर्देश | Usage Instructions:")
    print("  - टेक्स्ट दर्ज करें और एंटर दबाएं | Enter text and press Enter to speak (auto-detects language)")
    print("  - 'भाषा:टेक्स्ट' - विशिष्ट भाषा में बोलें | Speak in specific language")
    print("  - 'voice:भाषा:टेक्स्ट' - विशिष्ट आवाज़ और भाषा में बोलें | Speak with specific voice and language")
    print("  - 'save फाइलनाम.mp3' - ऑडियो को फाइल में सहेजें | Save audio to file")
    print("  - 'voices' - उपलब्ध आवाज़ें दिखाएं | Show available voices")
    print("  - 'config' - वर्तमान सेटिंग्स दिखाएं | Show current settings")
    print("  - 'quit' - समाप्त करें | Exit")
    print("\nउपलब्ध भाषाएं | Available languages:")
    for lang in IndianLanguage:
        print(f"  {lang.name.lower()}:{lang.value} ({lang.value})")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 अलविदा! | Goodbye!")
                break

            if not user_input:
                print("⚠️ कृपया टेक्स्ट दर्ज करें | Please enter text")
                continue

            if user_input.lower() == 'config':
                print(f"\n⚙️ वर्तमान सेटिंग्स | Current Settings:")
                print(f"  बैकेंड: {tts.backend}")
                print(f"  आवाज़: {config.voice_gender.value if not config.voice else config.voice}")
                print(f"  गति: {config.speech_rate} (edge_tts में डिफ़ॉल्ट का उपयोग किया जाता है | used as default in edge_tts)")
                print(f"  आवाज़ स्तर: {config.volume} (edge_tts में डिफ़ॉल्ट का उपयोग किया जाता है | used as default in edge_tts)")
                continue

            if user_input.lower() == 'voices':
                if tts.backend == "edge_tts":
                    voices = asyncio.run(list_available_voices())
                    print("\n🎙️ उपलब्ध आवाज़ें | Available Voices:")
                    for voice in voices:
                        print(f"  {voice['Name']} ({voice['Gender']}): {voice['Language']} - {voice['Description']}")
                else:
                    print("⚠️ आवाज़ सूची केवल edge_tts बैकेंड के लिए उपलब्ध है | Voice list only available for edge_tts backend")
                continue

            if user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or "output.mp3"
                if not filename.endswith(('.mp3', '.wav')):
                    filename += '.mp3'

                save_input = input("सहेजने के लिए टेक्स्ट दर्ज करें (voice:language:text या language:text या text) | Enter text to save (voice:language:text or language:text or text): ").strip()
                if not save_input:
                    print("⚠️ कोई टेक्स्ट दर्ज नहीं किया गया | No text entered")
                    continue

                voice = None
                language = IndianLanguage.HINDI
                save_text = save_input

                if save_input.count(':') == 2:
                    voice, lang_code, save_text = save_input.split(':', 2)
                    voice = voice.strip()
                    lang_code = lang_code.strip().lower()
                    try:
                        language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                    except StopIteration:
                        print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                        continue
                elif save_input.count(':') == 1:
                    lang_code, save_text = save_input.split(':', 1)
                    lang_code = lang_code.strip().lower()
                    try:
                        language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                    except StopIteration:
                        print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                        continue
                else:
                    lang_code = preprocessor.detect_language(save_text)
                    try:
                        language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                    except StopIteration:
                        print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                        continue

                try:
                    success, metrics = tts.save_to_file(save_text, filename, language, voice)
                    if success:
                        print(f"✓ ऑडियो सहेजा गया: {filename}")
                        print(metrics)
                    else:
                        print(f"❌ ऑडियो सहेजने में विफल | Failed to save audio")
                        print("कृपया सुनिश्चित करें कि ffmpeg इंस्टॉल है और नेटवर्क कनेक्शन स्थिर है | Ensure ffmpeg is installed and network is stable")
                except Exception as e:
                    print(f"❌ त्रुटि: {e} | Error: {e}")
                continue

            voice = None
            language = IndianLanguage.HINDI
            text = user_input

            if user_input.count(':') == 2:
                voice, lang_code, text = user_input.split(':', 2)
                voice = voice.strip()
                lang_code = lang_code.strip().lower()
                try:
                    language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                except StopIteration:
                    print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                    continue
            elif user_input.count(':') == 1:
                lang_code, text = user_input.split(':', 1)
                lang_code = lang_code.strip().lower()
                try:
                    language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                except StopIteration:
                    print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                    continue
            else:
                lang_code = preprocessor.detect_language(text)
                try:
                    language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                except StopIteration:
                    print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                    continue

            print(f"🔊 बोल रहा हूँ: {text[:50]}... ({language.name}, Voice: {voice if voice else tts._get_default_voice(language)})")

            start_time = time.time()
            success, metrics = asyncio.run(tts.speak(text, language, voice))
            elapsed = time.time() - start_time

            if success:
                print("✓ पूर्ण हुआ | Completed successfully")
                print(metrics)
                print(f"⏱️  कुल समय: {elapsed:.3f} सेकंड | Total time: {elapsed:.3f}s")
            else:
                print("❌ विफल | Failed")
                print(metrics)
                print("कृपया सुनिश्चित करें कि ffmpeg, ऑडियो डिवाइस, और नेटवर्क कनेक्शन उपलब्ध है | Ensure ffmpeg, audio device, and network are available")

        except KeyboardInterrupt:
            print("\n👋 अलविदा! | Goodbye!")
            break
        except Exception as e:
            print(f"❌ त्रुटि: {e} | Error: {e}")
            print("कृपया सुनिश्चित करें कि pygame, ffmpeg, और langdetect इंस्टॉल हैं | Ensure pygame, ffmpeg, and langdetect are installed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)