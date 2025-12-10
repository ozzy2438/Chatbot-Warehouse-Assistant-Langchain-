"""
AI Operations Assistant - Voice-Enabled Chatbot
================================================
Sesli e-ticaret envanter asistanı.

Tech Stack:
- Speech-to-Text: OpenAI Whisper API
- LLM: GPT-4o-mini + LangChain RAG
- Text-to-Speech: OpenAI TTS API

Usage:
    python voice_chatbot.py
"""

import os
import io
import tempfile
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

# Import our chatbot
from chatbot import ProductChatbot

# Load environment variables
load_dotenv()

# Initialize
console = Console()
client = OpenAI()

# Audio settings
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
CHANNELS = 1


class VoiceChatbot:
    """Voice-enabled AI Operations Assistant."""
    
    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._init_chatbot()
        self.is_recording = False
        self.audio_data = []
    
    def _check_api_key(self):
        """Check if OpenAI API key is set."""
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("[bold red]❌ OPENAI_API_KEY not found![/bold red]")
            raise SystemExit(1)
    
    def _init_chatbot(self):
        """Initialize the text chatbot."""
        self.console.print("[cyan]🎤 Initializing Voice Assistant...[/cyan]")
        self.chatbot = ProductChatbot()
        self.console.print("[green]✓ Voice Assistant ready![/green]")
    
    def record_audio(self, duration: float = None, silence_threshold: float = 0.01, 
                     silence_duration: float = 1.5) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Fixed duration in seconds (None for auto-stop on silence)
            silence_threshold: Volume threshold for silence detection
            silence_duration: Seconds of silence before stopping
        
        Returns:
            Audio data as numpy array
        """
        self.audio_data = []
        self.is_recording = True
        silence_counter = 0
        
        def callback(indata, frames, time, status):
            if status:
                self.console.print(f"[yellow]Audio status: {status}[/yellow]")
            
            self.audio_data.append(indata.copy())
            
            # Check for silence (auto-stop)
            if duration is None:
                volume = np.abs(indata).mean()
                nonlocal silence_counter
                if volume < silence_threshold:
                    silence_counter += frames / SAMPLE_RATE
                else:
                    silence_counter = 0
                
                if silence_counter >= silence_duration and len(self.audio_data) > 10:
                    self.is_recording = False
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                           callback=callback, dtype='float32'):
            
            if duration:
                # Fixed duration recording
                time.sleep(duration)
            else:
                # Auto-stop on silence
                start_time = time.time()
                max_duration = 30  # Maximum 30 seconds
                
                while self.is_recording and (time.time() - start_time) < max_duration:
                    time.sleep(0.1)
        
        if self.audio_data:
            return np.concatenate(self.audio_data, axis=0)
        return np.array([])
    
    def speech_to_text(self, audio_data: np.ndarray) -> str:
        """
        Convert speech to text using OpenAI Whisper.

        Args:
            audio_data: Audio as numpy array

        Returns:
            Transcribed text
        """
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            temp_path = f.name

        try:
            # Send to Whisper API
            with open(temp_path, "rb") as audio_file:
                # Add prompt with key terms to help Whisper recognize names
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="tr",  # Turkish
                    prompt="Sydney, Melbourne, Brisbane, Perth, Electronics, AirPods, iPhone, MacBook, iPad, Bluetooth, USB, Fire TV"
                )
            return transcript.text
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text: Text to convert
        
        Returns:
            Audio data as bytes
        """
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice="nova",   # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="mp3"
        )
        return response.content
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio from bytes."""
        # Save MP3 to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            mp3_path = f.name
        
        wav_path = None
        try:
            # Convert MP3 to WAV using pydub (soundfile can't read mp3 directly)
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(mp3_path)
            wav_path = mp3_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
            
            # Play WAV file
            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            sd.wait()  # Wait until playback is finished
            
        except Exception as e:
            self.console.print(f"[yellow]Playback error: {e}[/yellow]")
            # Fallback: try afplay on macOS
            try:
                import subprocess
                subprocess.run(["afplay", mp3_path], check=True)
            except:
                self.console.print("[red]Could not play audio. Install ffmpeg: brew install ffmpeg[/red]")
        finally:
            # Clean up temp files
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    def process_voice_query(self) -> tuple[str, str]:
        """
        Record voice, transcribe, get answer, and speak response.
        
        Returns:
            Tuple of (question, answer)
        """
        # Record
        self.console.print("\n[bold green]🎤 Dinliyorum... (konuşmayı bitirince sessiz kalın)[/bold green]")
        
        audio_data = self.record_audio()
        
        if len(audio_data) == 0:
            return None, "Ses kaydedilemedi."
        
        self.console.print("[cyan]📝 Yazıya çeviriyorum...[/cyan]")
        
        # Transcribe
        question = self.speech_to_text(audio_data)
        
        if not question.strip():
            return None, "Ses anlaşılamadı."
        
        self.console.print(f"\n[bold yellow]❓ Soru:[/bold yellow] {question}")
        
        # Get answer
        self.console.print("[cyan]🤔 Düşünüyorum...[/cyan]")
        answer = self.chatbot.ask(question)
        
        self.console.print(f"\n[bold green]🤖 Cevap:[/bold green]")
        self.console.print(Panel(answer, border_style="green"))
        
        # Speak the answer
        self.console.print("[cyan]🔊 Sesli yanıt hazırlanıyor...[/cyan]")
        
        # Truncate long answers for TTS (max ~500 chars)
        tts_text = answer[:500] + "..." if len(answer) > 500 else answer
        # Clean markdown for speech
        tts_text = tts_text.replace("**", "").replace("*", "").replace("#", "").replace("|", "")
        tts_text = tts_text.replace("---", "").replace("```", "")
        
        audio_response = self.text_to_speech(tts_text)
        
        self.console.print("[green]🔊 Konuşuyorum...[/green]")
        self.play_audio(audio_response)
        
        return question, answer
    
    def run_interactive(self):
        """Run interactive voice chat session."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]🎤 AI Operations Assistant - Voice Mode[/bold green]\n"
            "[dim]Sesli E-Ticaret Envanter Asistanı[/dim]\n\n"
            "Nasıl Kullanılır:\n"
            "• [cyan]ENTER[/cyan] tuşuna basın ve sorunuzu söyleyin\n"
            "• Konuşmayı bitirince sessiz kalın (otomatik durur)\n"
            "• Sistem sorunuzu yazıya çevirir ve sesli yanıt verir\n\n"
            "Örnek sorular:\n"
            "• 'Fire TV Stick'ten kaç tane var?'\n"
            "• 'Melbourne deposunun günlük geliri ne kadar?'\n"
            "• 'En çok satan ürünler hangileri?'\n\n"
            "[yellow]Çıkmak için 'q' yazın[/yellow]",
            title="🛒 Sesli Asistan",
            border_style="green"
        ))
        
        while True:
            try:
                self.console.print()
                action = Prompt.ask(
                    "[bold cyan]Soru sormak için ENTER'a basın (çıkış: q)[/bold cyan]",
                    default=""
                )
                
                if action.lower() in ['q', 'exit', 'quit', 'çık']:
                    self.console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
                    break
                
                # Process voice query
                self.process_voice_query()
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]👋 Görüşmek üzere![/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]❌ Hata: {e}[/red]")


class TextToVoiceChatbot:
    """Text input with voice output - for testing without microphone."""
    
    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._init_chatbot()
    
    def _check_api_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("[bold red]❌ OPENAI_API_KEY not found![/bold red]")
            raise SystemExit(1)
    
    def _init_chatbot(self):
        self.console.print("[cyan]🔊 Initializing Text-to-Voice Assistant...[/cyan]")
        self.chatbot = ProductChatbot()
        self.console.print("[green]✓ Assistant ready![/green]")
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech."""
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="mp3"
        )
        return response.content
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            mp3_path = f.name
        
        wav_path = None
        try:
            # Convert MP3 to WAV using pydub
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(mp3_path)
            wav_path = mp3_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
            
            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            self.console.print(f"[yellow]Playback error: {e}[/yellow]")
            # Fallback: afplay on macOS
            try:
                import subprocess
                subprocess.run(["afplay", mp3_path], check=True)
            except:
                self.console.print("[red]Could not play audio[/red]")
        finally:
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    def run_interactive(self):
        """Run text input with voice output."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]🔊 AI Operations Assistant - Text to Voice[/bold green]\n"
            "[dim]Yazılı soru, sesli cevap[/dim]\n\n"
            "Sorunuzu yazın, cevap sesli okunacak.\n\n"
            "[yellow]Çıkmak için 'q' yazın[/yellow]",
            title="🛒 Sesli Yanıt Modu",
            border_style="green"
        ))
        
        while True:
            try:
                self.console.print()
                question = Prompt.ask("[bold cyan]Soru[/bold cyan]")
                
                if question.lower() in ['q', 'exit', 'quit', 'çık']:
                    self.console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                # Get answer
                with self.console.status("[bold green]Düşünüyorum...[/bold green]"):
                    answer = self.chatbot.ask(question)
                
                self.console.print(Panel(answer, title="🤖 Cevap", border_style="green"))
                
                # Speak
                self.console.print("[cyan]🔊 Sesli yanıt...[/cyan]")
                tts_text = answer[:500] + "..." if len(answer) > 500 else answer
                tts_text = tts_text.replace("**", "").replace("*", "").replace("#", "").replace("|", "")
                
                audio = self.text_to_speech(tts_text)
                self.play_audio(audio)
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]👋 Görüşmek üzere![/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]❌ Hata: {e}[/red]")


def main():
    """Main entry point."""
    import sys
    
    console = Console()
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   AI Operations Assistant - Voice        ║[/bold blue]")
    console.print("[bold blue]║   Sesli E-Ticaret Asistanı               ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════════╝[/bold blue]")
    console.print()
    
    # Check for mode
    if "--text" in sys.argv:
        # Text input, voice output
        bot = TextToVoiceChatbot()
        bot.run_interactive()
    else:
        # Full voice mode
        try:
            bot = VoiceChatbot()
            bot.run_interactive()
        except Exception as e:
            console.print(f"[red]Voice mode error: {e}[/red]")
            console.print("[yellow]Trying text-to-voice mode instead...[/yellow]")
            bot = TextToVoiceChatbot()
            bot.run_interactive()


if __name__ == "__main__":
    main()

