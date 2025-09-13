#& "C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe" -m venv itv2
#code to create a virtual environment
#.\itv2\Scripts\Activate.ps1 to activate the virtual environment

import asyncio
import logging
import time
import tempfile
import os
import os.path
from typing import Tuple
from enum import Enum
from dataclasses import dataclass
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
import xml.sax.saxutils as saxutils

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
    """Supported Indian languages with language codes"""
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
    """Configuration for TTS system"""
    backend: str = "auto"
    voice_gender: VoiceGender = VoiceGender.FEMALE
    speech_rate: str = "medium"  # Not used in edge_tts without SSML
    volume: float = 0.9  # Not used in edge_tts without SSML
    timeout: float = 30.0

@dataclass
class LatencyMetrics:
    """Detailed latency tracking metrics"""
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
    """Enhanced text preprocessor optimized for Indian languages"""

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
            '18': 'अठारह', '19': 'उन्नीस', '20': 'बीस'
        }

    def clean_text(self, text: str) -> Tuple[str, float]:
        start_time = time.time()
        if not text.strip():
            return "", time.time() - start_time

        # Strip XML/SSML tags to extract plain text
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace and control characters
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)

        # Convert numbers to words (Hindi-specific)
        text = re.sub(r'\b\d+\b', lambda m: self._indian_number_to_words(m.group()), text)

        # Add pauses for punctuation
        text = re.sub(r'[.!?]', r'\g<0> ', text)
        text = re.sub(r'[,;:]', r'\g<0> ', text)

        # Escape special characters for safety
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

class IndianTTS:
    """Optimized TTS Engine for Indian Languages with detailed latency tracking"""

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

    async def _speak_async(self, text: str, language: IndianLanguage) -> Tuple[bool, LatencyMetrics]:
        metrics = LatencyMetrics()
        total_start = time.time()

        if not text.strip():
            logger.warning("Received empty text input")
            return False, metrics

        clean_text, text_processing_time = self.preprocessor.clean_text(text)
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
                success, playback_time = await self._speak_edge_tts(clean_text, language)
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

    def speak(self, text: str, language: IndianLanguage = IndianLanguage.HINDI) -> Tuple[bool, LatencyMetrics]:
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self._speak_async(text, language))
        except RuntimeError:
            return asyncio.run(self._speak_async(text, language))

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

    async def _speak_edge_tts(self, text: str, language: IndianLanguage) -> Tuple[bool, float]:
        try:
            import edge_tts
            import pygame

            if not self.pygame_initialized:
                self._initialize_pygame()

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
                IndianLanguage.ENGLISH: "en-IN-NeerjaNeural" if self.config.voice_gender == VoiceGender.FEMALE else "en-IN-PrabhatNeural"
            }

            voice = voice_map.get(language, None)
            if not voice:
                logger.warning(f"No voice available for {language.name}, defaulting to Hindi")
                voice = "hi-IN-SwaraNeural"

            # Use plain text directly with edge_tts
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                logger.error("No audio data received from Edge TTS")
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

    async def _save_edge_tts(self, text: str, output_path: str, language: IndianLanguage) -> bool:
        try:
            import edge_tts
            import os

            voice_map = {
                IndianLanguage.HINDI: "hi-IN-SwaraNeural",
                IndianLanguage.BENGALI: "bn-IN-TanishaaNeural",
                IndianLanguage.TAMIL: "ta-IN-PallaviNeural",
                IndianLanguage.TELUGU: "te-IN-ShrutiNeural",
                IndianLanguage.MARATHI: "mr-IN-AarohiNeural",
                IndianLanguage.GUJARATI: "gu-IN-DhwaniNeural",
                IndianLanguage.KANNADA: "kn-IN-SapnaNeural",
                IndianLanguage.MALAYALAM: "ml-IN-SobhanaNeural",
                IndianLanguage.PUNJABI: "pa-IN-KalpanaNeural",
                IndianLanguage.ENGLISH: "en-IN-NeerjaNeural"
            }

            voice = voice_map.get(language, None)
            if not voice:
                logger.warning(f"No voice available for {language.name}, defaulting to Hindi")
                voice = "hi-IN-SwaraNeural"

            # Use plain text directly
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                logger.error("No audio data received from Edge TTS")
                return False

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            return True

        except Exception as e:
            logger.error(f"Edge TTS save error: {e}")
            return False

    def save_to_file(self, text: str, output_path: str, language: IndianLanguage = IndianLanguage.HINDI) -> Tuple[bool, LatencyMetrics]:
        metrics = LatencyMetrics()
        total_start = time.time()

        clean_text, text_processing_time = self.preprocessor.clean_text(text)
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
                success = asyncio.run(self._save_edge_tts(clean_text, output_path, language))
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

def interactive_demo():
    print("\n🎤 भारतीय भाषा पाठ-से-वाणी प्रणाली | Indian Language Text-to-Speech System")
    print("=" * 70)

    if not TTS_BACKENDS:
        print("❌ कोई TTS बैकेंड उपलब्ध नहीं है! | No TTS backends available!")
        print("कृपया इंस्टॉल करें: pip install edge-tts gtts pygame")
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
        print(f"✓ TTS इंजन प्रारंभ किया गया | TTS engine initialized ({tts.backend})")
    except Exception as e:
        print(f"❌ TTS इंजन प्रारंभ करने में विफल | Failed to initialize TTS: {e}")
        print("कृपया सुनिश्चित करें कि pygame और ffmpeg इंस्टॉल हैं | Ensure pygame and ffmpeg are installed")
        return

    print("\n📝 उपयोग निर्देश | Usage Instructions:")
    print("  - टेक्स्ट दर्ज करें और एंटर दबाएं | Enter text and press Enter to speak")
    print("  - 'भाषा:टेक्स्ट' - विशिष्ट भाषा में बोलें | Speak in specific language")
    print("  - 'save फाइलनाम.mp3' - ऑडियो को फाइल में सहेजें | Save audio to file")
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
                print(f"  आवाज़: {config.voice_gender.value}")
                print(f"  गति: {config.speech_rate} (edge_tts में डिफ़ॉल्ट का उपयोग किया जाता है | used as default in edge_tts)")
                print(f"  आवाज़ स्तर: {config.volume} (edge_tts में डिफ़ॉल्ट का उपयोग किया जाता है | used as default in edge_tts)")
                continue

            if user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or "output.mp3"
                if not filename.endswith(('.mp3', '.wav')):
                    filename += '.mp3'

                save_text = input("सहेजने के लिए टेक्स्ट दर्ज करें | Enter text to save: ").strip()
                if not save_text:
                    print("⚠️ कोई टेक्स्ट दर्ज नहीं किया गया | No text entered")
                    continue

                try:
                    success, metrics = tts.save_to_file(save_text, filename)
                    if success:
                        print(f"✓ ऑडियो सहेजा गया: {filename}")
                        print(metrics)
                    else:
                        print(f"❌ ऑडियो सहेजने में विफल | Failed to save audio")
                        print("कृपया सुनिश्चित करें कि ffmpeg इंस्टॉल है | Ensure ffmpeg is installed")
                except Exception as e:
                    print(f"❌ त्रुटि: {e} | Error: {e}")
                continue

            if ':' in user_input:
                lang_part, text = user_input.split(':', 1)
                lang_code = lang_part.strip().lower()
                try:
                    language = next(lang for lang in IndianLanguage if lang.value == lang_code or lang.name.lower() == lang_code)
                except StopIteration:
                    print(f"❌ असमर्थित भाषा: {lang_code} | Unsupported language")
                    continue
            else:
                text = user_input
                language = IndianLanguage.HINDI

            print(f"🔊 बोल रहा हूँ: {text[:50]}... ({language.name})")

            start_time = time.time()
            success, metrics = tts.speak(text, language)
            elapsed = time.time() - start_time

            if success:
                print("✓ पूर्ण हुआ | Completed successfully")
                print(metrics)
                print(f"⏱️  कुल समय: {elapsed:.3f} सेकंड | Total time: {elapsed:.3f}s")
            else:
                print("❌ विफल | Failed")
                print(metrics)
                print("कृपया सुनिश्चित करें कि ffmpeg इंस्टॉल है और ऑडियो डिवाइस उपलब्ध है | Ensure ffmpeg is installed and audio device is available")

        except KeyboardInterrupt:
            print("\n👋 अलविदा! | Goodbye!")
            break
        except Exception as e:
            print(f"❌ त्रुटि: {e} | Error: {e}")
            print("कृपया सुनिश्चित करें कि ffmpeg और pygame इंस्टॉल हैं | Ensure ffmpeg and pygame are installed")

if __name__ == "__main__":
    interactive_demo()