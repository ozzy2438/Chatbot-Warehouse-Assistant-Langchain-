"""
Quick test script for the AI Operations Assistant
Run with: python test_chatbot.py
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file first
load_dotenv()

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("\n" + "="*60)
    print("❌ OPENAI_API_KEY not set!")
    print("="*60)
    print("\nPlease set your API key:")
    print("\n  Option 1 - Export in terminal:")
    print("    export OPENAI_API_KEY='sk-your-key-here'")
    print("\n  Option 2 - Create .env file:")
    print("    echo 'OPENAI_API_KEY=your-api-key-here' > .env")
    print("\n  Get your key from: https://platform.openai.com/api-keys")
    print("="*60 + "\n")
    sys.exit(1)

from chatbot import ProductChatbot

def main():
    print("\n" + "="*60)
    print("🧪 AI Operations Assistant - Test Mode")
    print("="*60 + "\n")
    
    # Initialize chatbot
    print("Initializing chatbot...\n")
    chatbot = ProductChatbot()
    
    # Test queries - including NEW LLM-driven intent routing tests
    test_questions = [
        # === Basic RAG queries ===
        "Fire TV Stick'ten kaç tane stokta var?",
        "En yüksek rating'li Electronics ürünü hangisi?",
        "AirPods alternatifi ne önerebiliriz?",

        # === NEW: Warehouse Recommendations (intent: recommendation) ===
        "Best warehouse for books if Brisbane is full",
        "Where should I send electronics if Sydney has no stock?",
        "Which city is best for Home & Kitchen products?",

        # === NEW: Simulations (intent: simulation) ===
        "If I remove 50 electronics from Melbourne, how many left?",
        "What if I add 100 books to Sydney warehouse?",
        "If customers order 10 items from Perth, how many remain?",

        # === NEW: Comparisons (intent: comparison) ===
        "Compare total items between Sydney and Perth",
        "What's the difference in stock between Melbourne and Brisbane?",

        # === NEW: Count Queries (intent: count) ===
        "How many electronics in Sydney?",
        "Total books across all warehouses?",
    ]

    print("\n" + "="*60)
    print("📋 Running Test Queries")
    print("="*60)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─'*60}")
        print(f"❓ Test {i}: {question}")
        print(f"{'─'*60}")

        try:
            answer = chatbot.ask(question)
            print(f"\n🤖 Answer:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        print()

    # Context/Follow-up tests
    print("\n" + "="*60)
    print("🔄 Running Context/Follow-up Tests")
    print("="*60)

    context_tests = [
        # Test 1: Category + Location follow-up
        ("Electronics kaç ürün var?", "İlk sorgu: Electronics ürün sayısı"),
        ("Peki Sydney'de kaç tane var?", "Takip: Sydney + Electronics (önceki kategori korunmalı)"),
        ("listele", "Takip: Son sorgunun sonuçlarını listele"),

        # Test 2: "ne kadar" belirsizliği
        ("Melbourne'da ne kadar?", "Test: 'ne kadar' = ürün sayısı mı, stok miktarı mı?"),
    ]

    for question, description in context_tests:
        print(f"\n{'─'*60}")
        print(f"📝 {description}")
        print(f"❓ Soru: {question}")
        print(f"{'─'*60}")

        try:
            answer = chatbot.ask(question)
            print(f"\n🤖 Answer:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        print()
    
    print("="*60)
    print("✅ Tests completed!")
    print("="*60)
    print("\nTo run interactive mode: python chatbot.py")
    print("To rebuild vector store: python chatbot.py --rebuild")


if __name__ == "__main__":
    main()

