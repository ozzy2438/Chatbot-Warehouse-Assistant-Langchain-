"""
AI Operations Assistant - Continuous Voice Mode
================================================
Sürekli dinleyen, doğal konuşma akışlı sesli asistan.
ChatGPT voice mode gibi - ENTER'sız, kesintisiz konuşma.

Usage:
    python voice_assistant.py
"""

import os
import io
import tempfile
import threading
import time
import queue
from pathlib import Path
from dotenv import load_dotenv

import re
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

# Import our chatbot
from chatbot import ProductChatbot

# Load environment variables
load_dotenv()

# Initialize
console = Console()
client = OpenAI()

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1


class ContinuousVoiceAssistant:
    """
    Sürekli dinleyen sesli asistan.
    - Konuşmayı algılar ve otomatik kaydeder
    - Sessizlik sonrası otomatik işler
    - Yüksek kaliteli TTS ile yanıt verir
    """
    
    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._init_chatbot()
        
        # State
        self.is_running = True
        self.is_listening = True
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        
        # Voice detection settings - IMPROVED for better detection
        self.silence_threshold = 0.01   # Lower = more sensitive to silence
        self.speech_threshold = 0.015   # Lower = easier to detect speech start
        self.silence_duration = 2.5     # Wait 2.5 seconds of silence before processing (was 1.8)
        self.min_speech_duration = 0.3  # Minimum speech to process (was 0.5)
        self.max_speech_duration = 30   # Maximum recording time
        
    def _check_api_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("[bold red]❌ OPENAI_API_KEY not found![/bold red]")
            raise SystemExit(1)
    
    def _init_chatbot(self):
        self.console.print("[cyan]🤖 Initializing AI Assistant...[/cyan]")
        self.chatbot = ProductChatbot()
        self.console.print("[green]✓ Ready to chat![/green]")
    
    def speech_to_text(self, audio_data: np.ndarray) -> str:
        """Convert speech to text using Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                # Add prompt with key terms to help Whisper recognize names
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="tr",
                    prompt="Sydney, Melbourne, Brisbane, Perth, Electronics, AirPods, iPhone, MacBook, iPad, Bluetooth, USB"
                )
            return transcript.text
        finally:
            os.unlink(temp_path)
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using high-quality TTS."""
        # Clean text for better speech - remove markdown formatting
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        clean_text = clean_text.replace("|", " ").replace("---", "").replace("```", "")
        clean_text = clean_text.replace("📊", "").replace("📋", "").replace("💡", "")
        clean_text = clean_text.replace("⭐", " yıldız ").replace("⚠️", " uyarı ")
        clean_text = clean_text.replace("🏆", "").replace("📈", "").replace("🔊", "")
        clean_text = clean_text.replace("💰", "").replace("📦", "")
        
        # Clean up whitespace and newlines - make it more natural for speech
        clean_text = re.sub(r'\n+', '. ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'\.+', '.', clean_text)  # Multiple dots to single
        clean_text = clean_text.strip()
        
        # Make numbers more speakable
        clean_text = clean_text.replace("$", " dolar ")
        clean_text = clean_text.replace("N/A", "bilgi yok")
        
        # Truncate if too long (TTS has limits) - but keep it shorter for better quality
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "... ve daha fazlası."
        
        # Use alloy voice - more natural sounding for Turkish
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",  # alloy is clearer and more natural
            input=clean_text,
            response_format="mp3",  # mp3 is more reliable
            speed=0.95  # Slightly slower for clarity
        )
        return response.content
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio response using macOS afplay (most reliable)."""
        self.is_speaking = True
        mp3_path = None
        try:
            # Save as MP3 (TTS returns MP3)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                mp3_path = f.name
            
            # Use macOS afplay - most reliable for MP3
            import subprocess
            subprocess.run(["afplay", mp3_path], check=True)
            
        except Exception as e:
            self.console.print(f"[red]Audio error: {e}[/red]")
        finally:
            self.is_speaking = False
            if mp3_path and os.path.exists(mp3_path):
                os.unlink(mp3_path)
    
    def record_until_silence(self) -> np.ndarray:
        """
        Record audio until user stops speaking.
        Returns audio data when silence is detected.
        Uses adaptive VAD for better speech detection.
        """
        audio_chunks = []
        silence_counter = 0
        speech_detected = False
        speech_duration = 0
        volume_history = []  # Track recent volumes for adaptive threshold
        
        def callback(indata, frames, time_info, status):
            nonlocal silence_counter, speech_detected, speech_duration, volume_history
            
            if self.is_speaking:
                return  # Don't record while speaking
            
            volume = np.abs(indata).mean()
            
            # Track volume history for adaptive threshold
            volume_history.append(volume)
            if len(volume_history) > 50:  # Keep last 50 samples (~5 seconds)
                volume_history.pop(0)
            
            # Use adaptive threshold based on background noise
            if len(volume_history) >= 10:
                noise_floor = np.percentile(volume_history, 20)  # 20th percentile as noise floor
                adaptive_threshold = max(self.speech_threshold, noise_floor * 2)
            else:
                adaptive_threshold = self.speech_threshold
            
            if volume > adaptive_threshold:
                speech_detected = True
                silence_counter = 0
                speech_duration += frames / SAMPLE_RATE
                audio_chunks.append(indata.copy())
            elif speech_detected:
                silence_counter += frames / SAMPLE_RATE
                audio_chunks.append(indata.copy())  # Keep recording during silence
                
                # Only stop after sustained silence
                if silence_counter >= self.silence_duration:
                    raise sd.CallbackStop()
        
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                               callback=callback, dtype='float32',
                               blocksize=int(SAMPLE_RATE * 0.05)):  # 50ms blocks (was 100ms) - more responsive
                
                # Wait for speech to complete
                start_time = time.time()
                while self.is_running and (time.time() - start_time) < self.max_speech_duration:
                    if not speech_detected:
                        time.sleep(0.02)  # Faster polling when waiting for speech
                    elif silence_counter >= self.silence_duration:
                        break
                    else:
                        time.sleep(0.02)
        except sd.CallbackStop:
            pass
        
        if audio_chunks and speech_duration >= self.min_speech_duration:
            return np.concatenate(audio_chunks, axis=0)
        return np.array([])
    
    def process_speech(self, audio_data: np.ndarray):
        """Process recorded speech and respond."""
        if len(audio_data) == 0:
            return
        
        # Transcribe
        self.console.print("\n[cyan]📝 Anlıyorum...[/cyan]")
        question = self.speech_to_text(audio_data)
        
        if not question.strip():
            self.console.print("[yellow]Anlaşılamadı, tekrar deneyin.[/yellow]")
            return
        
        # Check for exit commands
        if any(cmd in question.lower() for cmd in ['çıkış', 'kapat', 'görüşürüz', 'bye', 'exit']):
            self.console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
            self.is_running = False
            return
        
        self.console.print(f"\n[bold yellow]🎤 Siz:[/bold yellow] {question}")
        
        # Get answer
        self.console.print("[cyan]🤔 Düşünüyorum...[/cyan]")
        answer = self.chatbot.ask(question)
        
        self.console.print(f"\n[bold green]🤖 Asistan:[/bold green]")
        self.console.print(Panel(answer, border_style="green"))
        
        # Speak the answer
        self.console.print("[cyan]🔊 Konuşuyorum...[/cyan]")
        audio_response = self.text_to_speech(answer)
        self.play_audio(audio_response)
        
        self.console.print("\n[dim]🎤 Dinliyorum... (konuşmaya başlayın)[/dim]")
    
    def run(self):
        """Run continuous voice assistant."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]🤖 AI Operations Assistant[/bold green]\n"
            "[bold cyan]Sürekli Sesli Mod - ChatGPT Style[/bold cyan]\n\n"
            "[white]Doğrudan konuşmaya başlayın![/white]\n"
            "• Mikrofonunuz sürekli dinleniyor\n"
            "• Konuşmayı bitirince 2 saniye bekleyin\n"
            "• Sistem otomatik yanıt verecek\n\n"
            "[dim]Örnek sorular:[/dim]\n"
            "• 'Bluetooth kulaklıklardan kaç tane var?'\n"
            "• 'Sydney deposunda stok az mı?'\n"
            "• 'En çok satan ürünler hangileri?'\n\n"
            "[yellow]Çıkmak için 'çıkış' veya 'görüşürüz' deyin[/yellow]",
            title="🎙️ Sesli Asistan",
            border_style="cyan"
        ))
        
        self.console.print("\n[bold green]🎤 Dinliyorum... (konuşmaya başlayın)[/bold green]")
        
        while self.is_running:
            try:
                # Wait and record speech
                audio_data = self.record_until_silence()
                
                if len(audio_data) > 0:
                    self.process_speech(audio_data)
                    
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]👋 Görüşmek üzere![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Hata: {e}[/red]")
                time.sleep(1)


class PushToTalkAssistant:
    """
    Push-to-talk mode - Space tuşuna basılı tutarak konuşun.
    Daha kontrollü bir deneyim için.
    """
    
    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._init_chatbot()
        self.is_recording = False
        
    def _check_api_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(1)
    
    def _init_chatbot(self):
        self.console.print("[cyan]🤖 Initializing...[/cyan]")
        self.chatbot = ProductChatbot()
        self.console.print("[green]✓ Ready![/green]")
    
    def speech_to_text(self, audio_data: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            temp_path = f.name
        try:
            with open(temp_path, "rb") as audio_file:
                # Add prompt with key terms to help Whisper recognize names
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="tr",
                    prompt="Sydney, Melbourne, Brisbane, Perth, Electronics, AirPods, iPhone, MacBook, iPad, Bluetooth, USB"
                )
            return transcript.text
        finally:
            os.unlink(temp_path)
    
    def text_to_speech(self, text: str) -> bytes:
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        clean_text = clean_text.replace("|", "").replace("---", "")
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."
        
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=clean_text
        )
        return response.content
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio using macOS afplay."""
        mp3_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                mp3_path = f.name
            
            import subprocess
            subprocess.run(["afplay", mp3_path], check=True)
        except Exception as e:
            self.console.print(f"[red]Audio error: {e}[/red]")
        finally:
            if mp3_path and os.path.exists(mp3_path):
                os.unlink(mp3_path)
    
    def record_fixed_duration(self, duration: float = 5.0) -> np.ndarray:
        """Record for a fixed duration."""
        self.console.print(f"[green]🎤 Kayıt başladı ({duration}s)...[/green]")
        
        audio_data = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32'
        )
        sd.wait()
        
        return audio_data
    
    def run(self):
        """Run push-to-talk assistant."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]🤖 AI Operations Assistant[/bold green]\n"
            "[bold cyan]Push-to-Talk Mode[/bold cyan]\n\n"
            "• ENTER'a basın → 5 saniye konuşun\n"
            "• Sistem yanıt verecek\n\n"
            "[yellow]Çıkmak için 'q' yazın[/yellow]",
            title="🎙️ Sesli Asistan",
            border_style="cyan"
        ))
        
        while True:
            try:
                self.console.print("\n[bold cyan]ENTER'a basın ve konuşun (q=çıkış):[/bold cyan]")
                cmd = input()
                
                if cmd.lower() == 'q':
                    self.console.print("[yellow]👋 Görüşürüz![/yellow]")
                    break
                
                # Record
                audio_data = self.record_fixed_duration(5.0)
                
                # Transcribe
                self.console.print("[cyan]📝 Anlıyorum...[/cyan]")
                question = self.speech_to_text(audio_data)
                
                if not question.strip():
                    self.console.print("[yellow]Anlaşılamadı.[/yellow]")
                    continue
                
                self.console.print(f"\n[yellow]🎤 Siz:[/yellow] {question}")
                
                # Answer
                self.console.print("[cyan]🤔 Düşünüyorum...[/cyan]")
                answer = self.chatbot.ask(question)
                
                self.console.print(f"\n[green]🤖 Asistan:[/green]")
                self.console.print(Panel(answer, border_style="green"))
                
                # Speak
                self.console.print("[cyan]🔊 Konuşuyorum...[/cyan]")
                audio = self.text_to_speech(answer)
                self.play_audio(audio)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Görüşürüz![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Hata: {e}[/red]")


def main():
    import sys
    
    console = Console()
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   AI Operations Assistant - Voice 2.0   ║[/bold blue]")
    console.print("[bold blue]║   ChatGPT-Style Sesli Asistan           ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════════╝[/bold blue]")
    console.print()
    
    if "--ptt" in sys.argv:
        # Push-to-talk mode
        assistant = PushToTalkAssistant()
        assistant.run()
    else:
        # Continuous listening mode (default)
        assistant = ContinuousVoiceAssistant()
        assistant.run()


if __name__ == "__main__":
    main()

