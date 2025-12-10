"""
AI Operations Assistant - E-Commerce Inventory Chatbot
=======================================================
Natural language query system for e-commerce inventory and product management.

Tech Stack:
- Vector DB: ChromaDB (local, free)
- LLM: OpenAI GPT-4o-mini (cheap: ~$0.15/1M tokens)
- Framework: LangChain (RAG)
- Data: Pandas DataFrame for complex queries

Usage:
    python chatbot.py
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Constants
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "products"


class ProductChatbot:
    """AI-powered chatbot for e-commerce inventory queries."""
    
    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._load_dataframe()  # Load DataFrame for complex queries
        self._init_models()
        self._init_vectorstore()
        self._init_chains()
        
        # Conversation memory for context
        self.conversation_history = []
        self.last_query_result = None  # Store last query result for follow-ups

        # Context tracking for follow-up questions
        self.last_filters = {
            'location': None,
            'category': None,
            'product_keywords': [],
            'rating_filter': None,
            'price_filter': None,
            'stock_filter': None,
        }
        self.last_query_type = None  # 'count', 'list', 'summary'
    
    def _check_api_key(self):
        """Check if OpenAI API key is set."""
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("[bold red]❌ OPENAI_API_KEY not found![/bold red]")
            self.console.print("\n[yellow]Please set your API key:[/yellow]")
            self.console.print("  export OPENAI_API_KEY='your-key-here'")
            self.console.print("\n[dim]Or create a .env file with:[/dim]")
            self.console.print("  OPENAI_API_KEY=your-key-here")
            raise SystemExit(1)
    
    def _load_dataframe(self):
        """Load product data into DataFrame for complex queries."""
        self.console.print("[cyan]📊 Loading product database...[/cyan]")
        
        csv_path = DATA_DIR / "products_for_ai.csv"
        self.df = pd.read_csv(csv_path)
        
        # Also load alternatives
        alt_path = DATA_DIR / "products_with_alternatives.csv"
        df_alt = pd.read_csv(alt_path)
        self.df = self.df.merge(
            df_alt[['asin', 'similar_price_alternatives', 'higher_rated_alternatives']],
            on='asin',
            how='left'
        )
        
        self.console.print(f"[green]✓ Loaded {len(self.df)} products into memory[/green]")
    
    def _init_models(self):
        """Initialize LLM and embeddings."""
        self.console.print("[cyan]🔧 Initializing models...[/cyan]")
        
        # GPT-4o-mini - very cheap and fast
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )
        
        # OpenAI embeddings for vector search
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        self.console.print("[green]✓ Models initialized[/green]")
    
    def _load_product_documents(self) -> list[Document]:
        """Load product data and convert to LangChain documents."""
        self.console.print("[cyan]📦 Creating vector documents...[/cyan]")
        
        documents = []
        for _, row in self.df.iterrows():
            content = self._create_document_content(row)
            metadata = {
                "asin": str(row['asin']),
                "product_name": str(row['product_name']),
                "category": str(row['category']),
                "price": float(row['price']) if pd.notna(row['price']) else 0.0,
                "rating": float(row['rating']) if pd.notna(row['rating']) else 0.0,
                "stock_quantity": int(row['stock_quantity']) if pd.notna(row['stock_quantity']) else 0,
                "warehouse_location": str(row['warehouse_location']),
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        self.console.print(f"[green]✓ Created {len(documents)} documents[/green]")
        return documents
    
    def _create_document_content(self, row: pd.Series) -> str:
        """Create searchable document content from product row."""
        parts = [
            f"Product: {row['product_name']}",
            f"ASIN: {row['asin']}",
            f"Category: {row['category']}",
            f"Price: ${row['price']:.2f}" if pd.notna(row['price']) else "Price: Not available",
            f"Rating: {row['rating']}/5 stars ({row['review_count']:.0f} reviews)" if pd.notna(row['rating']) else "Rating: Not available",
            f"Stock: {row['stock_quantity']} units in {row['warehouse_location']} warehouse",
            f"Satisfaction: {row['satisfaction_level']}",
            f"Minimum Stock Threshold: {row['minimum_stock_threshold']}",
        ]
        
        if row['stock_quantity'] < row['minimum_stock_threshold']:
            parts.append(f"⚠️ LOW STOCK ALERT: Below minimum threshold of {row['minimum_stock_threshold']}")
            parts.append(f"Restock Date: {row['restock_date']}")
        
        if pd.notna(row.get('similar_price_alternatives')) and row.get('similar_price_alternatives'):
            parts.append(f"Similar Price Alternatives (ASINs): {row['similar_price_alternatives']}")
        
        if pd.notna(row.get('higher_rated_alternatives')) and row.get('higher_rated_alternatives'):
            parts.append(f"Higher Rated Alternatives (ASINs): {row['higher_rated_alternatives']}")
        
        return "\n".join(parts)
    
    def _init_vectorstore(self):
        """Initialize or load ChromaDB vector store."""
        self.console.print("[cyan]🗄️ Setting up vector database...[/cyan]")
        
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            self.console.print("[yellow]  Loading existing vector store...[/yellow]")
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            self.console.print("[green]✓ Vector store loaded[/green]")
        else:
            self.console.print("[yellow]  Creating new vector store (first run)...[/yellow]")
            documents = self._load_product_documents()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )
            splits = text_splitter.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )
            self.console.print(f"[green]✓ Vector store created with {len(splits)} chunks[/green]")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Increased from 5 to 10
        )
    
    def _init_chains(self):
        """Initialize the RAG and SQL chains."""
        self.console.print("[cyan]⛓️ Building query chains...[/cyan]")

        # RAG chain for product-specific queries
        rag_prompt = ChatPromptTemplate.from_template("""You are an AI Factory Manager and intelligent inventory advisor.
You help users manage warehouse operations with smart recommendations.

Use the following product information to answer questions:

{context}

BUSINESS INTELLIGENCE RULES:
1. **Stock Recommendations**: When asked "where should I go" or "which city", analyze ALL warehouses and recommend the one with HIGHEST stock
2. **Transfer Suggestions**: If one warehouse is low, suggest transferring from warehouses with excess stock
3. **Proactive Alerts**: Warn about low stock situations and suggest actions
4. **Alternative Locations**: Always provide backup options (2nd and 3rd best locations)
5. **Quantity Analysis**: Calculate totals across multiple locations when asked
6. **Business Context**: Consider stock levels, not just availability - recommend locations with SURPLUS stock

RESPONSE FORMAT:
- For "which city/where should I" questions:
  ✓ Primary Recommendation: [City with MOST stock] - [X items available]
  ✓ Alternative Options: [2nd best city], [3rd best city]
  ✓ Reason: "This location has the highest inventory for this category"

- For shortage scenarios:
  ⚠️ Alert: [Location] is low/out of stock
  💡 Suggestion: Transfer from [City with excess] ([Y items available])

- Show prices in USD with $ symbol
- ALWAYS answer in ENGLISH
- Be proactive and suggest actions, don't just state facts

User Question: {question}

Answer:""")
        
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        # Intent extraction chain - LLM understands ANY question format
        intent_prompt = ChatPromptTemplate.from_template("""You are an intent extraction system for inventory management queries.

Analyze this question and extract structured information in JSON format.

Question: {question}

Extract and return ONLY valid JSON (no markdown, no code blocks):
{{
  "intent": "recommendation|count|comparison|simulation|product_query|unknown",
  "locations": ["Sydney", "Melbourne"],
  "exclude_locations": ["Perth"],
  "category": "Electronics|Books|Home & Kitchen|Clothing, Shoes & Jewelry|null",
  "operation": "add|subtract|multiply|divide|null",
  "quantity": 100,
  "filters": {{"rating": ">4.5", "price": "<50"}},
  "product_keywords": ["ipad", "macbook"]
}}

Intent types:
- "recommendation": User wants warehouse/location suggestions (where should I go, which city, alternatives)
- "count": User wants to know how many items (how many, count, total)
- "comparison": User wants to compare locations/products (difference, compare)
- "simulation": User asks "what if" scenarios (if I order X, what happens)
- "product_query": User asks about specific products (which product is cheapest, most expensive)
- "unknown": Cannot determine intent

Rules:
1. If user says "if X has no Y" or "if X is out of stock", add X to exclude_locations
2. Extract ALL mentioned cities: Sydney, Melbourne, Brisbane, Perth
3. Category must match exactly: Electronics, Books, Home & Kitchen, or Clothing, Shoes & Jewelry
4. For "order 100 items", "subtract 100", etc., set operation="subtract" and quantity=100
5. Return ONLY the JSON, no other text

Answer:""")

        self.intent_chain = (
            intent_prompt
            | self.llm
            | JsonOutputParser()
        )

        self.console.print("[green]✓ Query chains ready (with LLM intent extraction)[/green]")
    
    def _get_database_summary(self) -> str:
        """Get summary statistics from the database."""
        summary_parts = []
        
        # Category statistics
        category_stats = self.df.groupby('category').agg({
            'asin': 'count',
            'rating': 'mean',
            'price': 'mean',
            'stock_quantity': 'sum'
        }).round(2)
        
        summary_parts.append("=== KATEGORİ İSTATİSTİKLERİ ===")
        for cat, row in category_stats.iterrows():
            summary_parts.append(f"{cat}: {int(row['asin'])} ürün, Ort. Rating: {row['rating']:.2f}, Ort. Fiyat: ${row['price']:.2f}, Toplam Stok: {int(row['stock_quantity'])}")
        
        # Warehouse statistics
        warehouse_stats = self.df.groupby('warehouse_location').agg({
            'asin': 'count',
            'stock_quantity': 'sum'
        })
        
        summary_parts.append("\n=== DEPO İSTATİSTİKLERİ ===")
        for loc, row in warehouse_stats.iterrows():
            summary_parts.append(f"{loc}: {int(row['asin'])} ürün, Toplam Stok: {int(row['stock_quantity'])}")
        
        # Low stock count
        low_stock = self.df[self.df['stock_quantity'] < self.df['minimum_stock_threshold']]
        summary_parts.append(f"\n=== DÜŞÜK STOK UYARISI ===")
        summary_parts.append(f"Toplam {len(low_stock)} ürün minimum stok seviyesinin altında")
        
        return "\n".join(summary_parts)
    
    def _get_smart_warehouse_recommendation(self, category: str = None, product_keywords: list = None, exclude_location: str = None) -> str:
        """Intelligent warehouse recommendation based on stock levels."""
        filtered_df = self.df.copy()

        # Apply filters
        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        if product_keywords:
            pattern = '|'.join(product_keywords)
            filtered_df = filtered_df[filtered_df['product_name'].str.lower().str.contains(pattern, na=False)]

        if exclude_location:
            filtered_df = filtered_df[filtered_df['warehouse_location'] != exclude_location]

        if len(filtered_df) == 0:
            return "⚠️ No stock found in any warehouse for this item."

        # Calculate stock by warehouse
        warehouse_stocks = filtered_df.groupby('warehouse_location').agg({
            'stock_quantity': 'sum',
            'asin': 'count'
        }).sort_values('stock_quantity', ascending=False)

        # Build recommendation
        lines = []

        if len(warehouse_stocks) == 0:
            return "⚠️ No alternative warehouses available."

        # Primary recommendation
        best_warehouse = warehouse_stocks.index[0]
        best_stock = int(warehouse_stocks.iloc[0]['stock_quantity'])
        best_items = int(warehouse_stocks.iloc[0]['asin'])

        lines.append(f"✅ **Primary Recommendation: {best_warehouse}**")
        lines.append(f"   - Stock Available: **{best_stock} units** ({best_items} different items)")
        lines.append(f"   - Reason: Highest inventory for this category")

        # Alternative options
        if len(warehouse_stocks) > 1:
            lines.append("\n📍 **Alternative Options:**")
            for i in range(1, min(len(warehouse_stocks), 3)):
                alt_warehouse = warehouse_stocks.index[i]
                alt_stock = int(warehouse_stocks.iloc[i]['stock_quantity'])
                alt_items = int(warehouse_stocks.iloc[i]['asin'])
                lines.append(f"   {i}. {alt_warehouse}: {alt_stock} units ({alt_items} items)")

        # Add context about excluded location if provided
        if exclude_location:
            excluded_stock = self.df[
                (self.df['warehouse_location'] == exclude_location) &
                (self.df['category'] == category if category else True)
            ]['stock_quantity'].sum()

            if excluded_stock == 0:
                lines.append(f"\n⚠️ **Alert:** {exclude_location} is OUT OF STOCK for this category")
                lines.append(f"💡 **Suggestion:** Transfer {min(best_stock // 2, 50)} units from {best_warehouse} to {exclude_location}")
            else:
                lines.append(f"\nℹ️ {exclude_location} has {int(excluded_stock)} units (lower than {best_warehouse})")

        return "\n".join(lines)

    def _simulate_scenario(self, intent: dict) -> str:
        """Handle 'what if' simulation scenarios."""
        operation = intent.get('operation')
        quantity = intent.get('quantity', 0)
        locations = intent.get('locations', [])
        category = intent.get('category')

        if not operation or not quantity:
            return "⚠️ I need to know the operation and quantity for simulation. Example: 'If I order 100 items from Sydney'"

        # Get current total
        filtered_df = self.df.copy()
        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]
        if locations:
            filtered_df = filtered_df[filtered_df['warehouse_location'].isin(locations)]

        current_total = len(filtered_df)

        # Simulate the operation
        if operation == 'subtract':
            # Check if operation is valid
            if quantity > current_total:
                return f"⚠️ **Cannot remove {quantity} items** - only **{current_total} items** available!\n\n" \
                       f"📍 Location: {', '.join(locations) if locations else 'All warehouses'}\n" \
                       f"📦 Category: {category if category else 'All categories'}"

            remaining = current_total - quantity
            lines = [
                f"📊 **Simulation: Subtract {quantity} items**",
                f"",
                f"• Current total: **{current_total} items**",
                f"• After removing {quantity}: **{remaining} items**",
                f"• Change: **-{quantity} items** ({-quantity/current_total*100:.1f}%)"
            ]

            if locations:
                lines.append(f"\n🏙️ Locations: {', '.join(locations)}")
            if category:
                lines.append(f"📦 Category: {category}")

            return "\n".join(lines)

        elif operation == 'add':
            new_total = current_total + quantity

            # Calculate percentage change (handle zero case)
            if current_total > 0:
                percent_change = f"(+{quantity/current_total*100:.1f}%)"
            else:
                percent_change = "(new inventory)"

            lines = [
                f"📊 **Simulation: Add {quantity} items**",
                f"",
                f"• Current total: **{current_total} items**",
                f"• After adding {quantity}: **{new_total} items**",
                f"• Change: **+{quantity} items** {percent_change}"
            ]

            if locations:
                lines.append(f"\n🏙️ Locations: {', '.join(locations)}")
            if category:
                lines.append(f"📦 Category: {category}")

            return "\n".join(lines)

        return "⚠️ Unsupported operation. Use 'add' or 'subtract'."

    def _parse_transfer_intent(self, question: str) -> dict:
        """Use LLM to parse complex multi-step transfer queries."""
        transfer_prompt = ChatPromptTemplate.from_template("""You are a transfer intent parser for inventory management.

Analyze this question and extract ALL transfer operations:

Question: {question}

Return ONLY valid JSON (no markdown):
{{
  "is_transfer": true,
  "transfers": [
    {{
      "from_city": "Sydney",
      "to_city": "Perth", 
      "category": "Electronics",
      "quantity": 300
    }},
    {{
      "from_city": "Melbourne",
      "to_city": "Brisbane",
      "category": "Books", 
      "quantity": 150
    }}
  ]
}}

Rules:
1. Extract ALL transfers mentioned (can be 1 or more)
2. Cities: Sydney, Melbourne, Brisbane, Perth
3. Categories: Electronics, Books, Home & Kitchen, Clothing, Sports & Outdoors, etc.
4. If no transfer detected, set "is_transfer": false and "transfers": []
5. Return ONLY the JSON

Answer:""")

        chain = transfer_prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"question": question})
            return result
        except Exception as e:
            return {"is_transfer": False, "transfers": [], "error": str(e)}

    def _execute_transfer_simulation(self, transfers: list) -> str:
        """Execute multi-step transfer calculations with real data."""
        if not transfers:
            return "⚠️ No valid transfers detected."

        lines = ["📦 **INVENTORY TRANSFER SIMULATION**", ""]
        
        # Track cumulative changes per city/category
        changes = {}  # {(city, category): delta}
        
        for i, transfer in enumerate(transfers, 1):
            from_city = transfer.get('from_city')
            to_city = transfer.get('to_city')
            category = transfer.get('category')
            quantity = transfer.get('quantity', 0)
            
            if not all([from_city, to_city, category, quantity]):
                lines.append(f"⚠️ Transfer {i}: Missing information, skipped.")
                continue
            
            # Get current stock levels from DataFrame
            from_stock = self.df[
                (self.df['warehouse_location'] == from_city) & 
                (self.df['category'] == category)
            ]['stock_quantity'].sum()
            
            to_stock = self.df[
                (self.df['warehouse_location'] == to_city) & 
                (self.df['category'] == category)
            ]['stock_quantity'].sum()
            
            # Track changes
            key_from = (from_city, category)
            key_to = (to_city, category)
            
            if key_from not in changes:
                changes[key_from] = {'original': int(from_stock), 'delta': 0}
            if key_to not in changes:
                changes[key_to] = {'original': int(to_stock), 'delta': 0}
            
            changes[key_from]['delta'] -= quantity
            changes[key_to]['delta'] += quantity
            
            # Format this transfer
            lines.append(f"**Transfer {i}:** {quantity:,} {category}")
            lines.append(f"  📤 From: {from_city} → 📥 To: {to_city}")
            lines.append("")
        
        # Calculate and show final results
        lines.append("---")
        lines.append("📊 **RESULTS AFTER ALL TRANSFERS:**")
        lines.append("")
        
        for (city, category), data in sorted(changes.items()):
            original = data['original']
            delta = data['delta']
            final = original + delta
            
            if delta < 0:
                arrow = "⬇️"
                change_str = f"{delta:,}"
            elif delta > 0:
                arrow = "⬆️"
                change_str = f"+{delta:,}"
            else:
                arrow = "➡️"
                change_str = "0"
            
            lines.append(f"**{city}** ({category}):")
            lines.append(f"  • Before: {original:,} units")
            lines.append(f"  • After: **{final:,} units** {arrow} ({change_str})")
            lines.append("")
            
            # Warning if negative stock
            if final < 0:
                lines.append(f"  ⚠️ **WARNING:** Insufficient stock! Need {abs(final):,} more units.")
                lines.append("")
        
        return "\n".join(lines)

    def _check_for_transfer_query(self, question: str) -> str:
        """Check if question is a transfer query and handle it."""
        question_lower = question.lower()
        
        # Keywords that indicate transfer intent
        transfer_keywords = [
            'transfer', 'move', 'send', 'ship', 'relocate',
            'from.*to', 'taşı', 'gönder', 'aktar'
        ]
        
        is_transfer = any(
            re.search(kw, question_lower) for kw in transfer_keywords
        )
        
        if not is_transfer:
            return None
        
        # Parse the transfer intent using LLM
        intent = self._parse_transfer_intent(question)
        
        if intent.get('is_transfer') and intent.get('transfers'):
            return self._execute_transfer_simulation(intent['transfers'])
        
        return None


    def _execute_data_query(self, question: str) -> str:
        """Execute complex data queries using reliable pattern matching."""
        question_lower = question.lower()

        # Check for smart recommendation queries
        recommendation_patterns = [
            'which city should', 'which warehouse should', 'where should i go',
            'where should i send', 'where should i',  # Added for "where should I send"
            'which location', 'recommend', 'suggest', 'alternative',
            'out of stock', 'low stock', 'no stock', 'has no'  # Added "has no"
        ]

        if any(pattern in question_lower for pattern in recommendation_patterns):
            # Extract context from question
            category = None
            for cat_key, cat_val in {
                'electronics': 'Electronics', 'clothing': 'Clothing, Shoes & Jewelry',
                'books': 'Books', 'home': 'Home & Kitchen'
            }.items():
                if cat_key in question_lower:
                    category = cat_val
                    break

            # Extract excluded location - COMPREHENSIVE DETECTION
            exclude_loc = None
            for loc in ['Melbourne', 'Sydney', 'Brisbane', 'Perth']:
                loc_lower = loc.lower()
                # Check multiple patterns for exclusion
                if loc_lower in question_lower:
                    # DEBUG
                    import logging
                    logging.info(f"🔍 Checking exclusion patterns for '{loc}' in question: '{question_lower}'")
                    # Pattern 1: "if X has no/zero" - MUST BE FIRST
                    if re.search(rf'if\s+{loc_lower}\s+has\s+(no|zero|low)', question_lower):
                        logging.info(f"✅ Matched Pattern 1: 'if {loc} has no/zero'")
                        exclude_loc = loc
                        break
                    # Pattern 2: "if X is out/no stock"
                    if re.search(rf'if\s+{loc_lower}\s+(is\s+)?(out|no\s+stock|low\s+stock)', question_lower):
                        exclude_loc = loc
                        break
                    # Pattern 3: "where...if X has" (order matters!)
                    if re.search(rf'where.*if\s+{loc_lower}\s+has', question_lower):
                        exclude_loc = loc
                        break
                    # Pattern 4: "where...if X" (general if clause)
                    if re.search(rf'where.*if\s+{loc_lower}', question_lower):
                        exclude_loc = loc
                        break
                    # Pattern 5: "X out of stock", "X has no stock"
                    if re.search(rf'{loc_lower}\s+(is\s+)?(out\s+of\s+stock|has\s+no|no\s+stock|low\s+stock)', question_lower):
                        exclude_loc = loc
                        break
                    # Pattern 6: "alternative to X", "instead of X"
                    if re.search(rf'(alternative\s+to|instead\s+of)\s+{loc_lower}', question_lower):
                        exclude_loc = loc
                        break

            # Use last filters if available
            if not category and self.last_filters.get('category'):
                category = self.last_filters['category']

            product_keywords = self._extract_product_keywords(question) or self.last_filters.get('product_keywords')

            return self._get_smart_warehouse_recommendation(category, product_keywords, exclude_loc)

        # Check for PRODUCT-SPECIFIC questions that need LLM reasoning (not location recommendations)
        # IMPORTANT: Don't fall back to RAG for warehouse/location questions - we handle those above
        product_specific_indicators = [
            'most expensive', 'cheapest', 'highest price', 'lowest price',
            'top rated', 'best rated', 'worst rated',
            'which product', 'which item', 'what product',
            'compare products', 'difference between products',
        ]

        # Only fall back to RAG if it's a product-specific query (not location/warehouse)
        is_product_query = any(indicator in question_lower for indicator in product_specific_indicators)
        is_location_query = any(word in question_lower for word in ['city', 'warehouse', 'location', 'where should', 'recommend warehouse', 'suggest warehouse'])

        if is_product_query and not is_location_query:
            return None  # Fall back to RAG for product-specific reasoning

        # Check for follow-up questions (only if no specific product/category mentioned)
        follow_up_keywords = ['bunlar', 'bu ürünler', 'bunları', 'onları', 'these', 'those', 'them']
        product_keywords = self._extract_product_keywords(question)
        has_category = any(cat in question_lower for cat in ['electronics', 'elektronik', 'books', 'kitap', 'home', 'ev', 'fashion', 'clothing', 'clothes'])
        has_location = any(loc in question_lower for loc in ['melbourne', 'sydney', 'brisbane', 'perth'])

        is_pure_follow_up = (any(kw in question_lower for kw in follow_up_keywords) and
                            not product_keywords and not has_category and not has_location)

        if is_pure_follow_up and self.last_query_result is not None:
            return self._handle_follow_up(question)

        # Check if this is a SIMPLE data/analytics question (counting, listing)
        data_keywords = ['how many', 'kaç', 'toplam', 'total', 'count', 'liste', 'list',
                        'show', 'göster', 'display', 'give me']

        location_keywords = ['melbourne', 'sydney', 'brisbane', 'perth']

        is_data_query = any(kw in question_lower for kw in data_keywords) or \
                        any(loc in question_lower for loc in location_keywords)

        if not is_data_query:
            return None  # Fall back to RAG

        # Detect ambiguous "ne kadar" questions and clarify intent
        # "ne kadar" can mean "how many products" OR "how much stock"
        # Default to product count unless explicitly asking for stock quantity
        is_stock_sum_question = any(phrase in question_lower for phrase in [
            'toplam stok', 'stok miktarı', 'kaç adet stok', 'stok toplamı',
            'ne kadar stok', 'total stock', 'stock quantity'
        ])

        try:
            # Use reliable pattern matching first
            result = self._reliable_data_query(question)
            if result:
                return result
            # Fall back to LLM-generated code
            return self._smart_data_query(question)
        except Exception as e:
            self.console.print(f"[dim]Query error: {e}, falling back to RAG[/dim]")
            return None
    
    def _is_follow_up_question(self, question_lower: str) -> bool:
        """Detect if this is a follow-up question that should inherit context."""
        # Follow-up indicators
        follow_up_patterns = [
            # "Peki X'de?" style questions
            r'^peki\s',
            r'^peki$',
            r'^ya\s',
            r'^e\s',
            r'^ama\s',
            # Direct location-only questions (implies keep other filters)
            r"sydney'de",
            r"sydney de",
            r"melbourne'da",
            r"melbourne da",
            r"brisbane'de",
            r"brisbane de", 
            r"perth'te",
            r"perth te",
            # Listing commands
            r'^listele',
            r'^göster',
            r'^bunları',
            r'^liste$',
            r'^listede$',
            # Reference to previous results
            r'onlar',
            r'bunlar',
            r'o ürünler',
            r'bu ürünler',
            r'sadece',  # "sadece elektronik" gibi
            # Short questions
            r'^ne kadar',
            r'^kaç tane',
        ]

        for pattern in follow_up_patterns:
            if re.search(pattern, question_lower):
                return True

        # Short questions with only location change are likely follow-ups
        # e.g., "Sydney'de kaç tane?" after "Electronics kaç tane?"
        words = question_lower.split()
        if len(words) <= 6:
            # Check if we have previous context
            has_previous_context = (self.last_filters.get('category') or 
                                   self.last_filters.get('product_keywords') or
                                   self.last_filters.get('location'))
            if has_previous_context:
                # Check if question mentions a location but no category
                has_location = any(loc in question_lower for loc in ['melbourne', 'sydney', 'brisbane', 'perth'])
                has_category = any(cat in question_lower for cat in ['electronics', 'elektronik', 'books', 'kitap', 'home', 'ev'])
                if has_location and not has_category:
                    return True
                # Or if it's just asking "kaç tane" type questions
                if any(q in question_lower for q in ['kaç', 'ne kadar', 'listele', 'göster']):
                    return True

        return False

    def _handle_follow_up(self, question: str) -> str:
        """Handle follow-up questions using last query result."""
        if self.last_query_result is None or 'df' not in self.last_query_result:
            return None

        result_df = self.last_query_result['df']
        query_desc = self.last_query_result.get('description', 'Önceki sorgu')

        # Check if user wants a specific number
        limit_match = re.search(r'(\d+)\s*(?:tane|adet|ürün|tanesini)', question.lower())
        limit = int(limit_match.group(1)) if limit_match else 10

        return self._format_product_list(result_df, query_desc, limit)
    
    def _breakdown_by_locations(self, question: str, specific_locations: list = None) -> str:
        """Break down counts by warehouse locations - all or specific ones."""
        # Use last query's filters if available
        base_df = self.df.copy()

        # Apply category filter if we have one from context
        if self.last_filters.get('category'):
            base_df = base_df[base_df['category'] == self.last_filters['category']]

        # Apply product keywords if we have them from context
        if self.last_filters.get('product_keywords'):
            pattern = '|'.join(self.last_filters['product_keywords'])
            base_df = base_df[base_df['product_name'].str.lower().str.contains(pattern, na=False)]

        # Count by location
        location_counts = base_df.groupby('warehouse_location').size().to_dict()

        # Determine which locations to show
        if specific_locations and len(specific_locations) >= 2:
            # Only show specified locations
            locations_to_show = specific_locations
            lines = [f"Total items in {' and '.join(specific_locations)}:", ""]
        else:
            # Show all locations (breakdown)
            locations_to_show = ['Sydney', 'Melbourne', 'Brisbane', 'Perth']
            lines = ["Breakdown by warehouse:", ""]

        # Show counts for selected locations
        total = 0
        for location in locations_to_show:
            count = location_counts.get(location, 0)
            lines.append(f"• {location}: **{count} items**")
            total += count

        lines.append(f"\n**Total: {total} items**")

        return "\n".join(lines)
    
    
    def _reliable_data_query(self, question: str) -> str:
        """Use reliable pattern matching for common query types."""
        question_lower = question.lower()

        # Special case: Multi-location breakdown or specific location total
        # "how many in Sydney, Melbourne and Perth" or "total with Melbourne and Sydney"
        multi_loc_patterns = [
            r'(?:sydney|melbourne|brisbane|perth).*(?:and|,|with).*(?:sydney|melbourne|brisbane|perth)',  # X and Y
            r'(?:report|breakdown|distribution|total).*(?:sydney|melbourne|brisbane|perth)',  # report for locations
        ]

        if any(re.search(pattern, question_lower) for pattern in multi_loc_patterns):
            # Extract mentioned locations
            mentioned_locations = []
            for loc in ['Sydney', 'Melbourne', 'Brisbane', 'Perth']:
                if loc.lower() in question_lower:
                    mentioned_locations.append(loc)

            # If 2+ specific locations mentioned, show only those
            if len(mentioned_locations) >= 2:
                return self._breakdown_by_locations(question_lower, mentioned_locations)
            # Otherwise show all locations breakdown
            elif len(mentioned_locations) >= 1:
                return self._breakdown_by_locations(question_lower)

        # Detect if this is a follow-up question
        is_follow_up = self._is_follow_up_question(question_lower)

        # Build filters from question
        filtered_df = self.df.copy()
        filters_applied = []
        sort_info = None

        # Current filters (will inherit from last_filters if follow-up)
        current_filters = {
            'location': None,
            'category': None,
            'product_keywords': [],
            'rating_filter': None,
            'price_filter': None,
            'stock_filter': None,
        }

        # Location filter - check for new location in question
        location = None
        location_mentioned = False
        for loc in ['melbourne', 'sydney', 'brisbane', 'perth']:
            if loc in question_lower:
                location = loc.title()
                location_mentioned = True
                break

        # Special case: If user asks "elimizde" or "toplam", clear location filter
        if any(word in question_lower for word in ['elimizde', 'toplam', 'tüm', 'all']):
            location = None
            location_mentioned = True  # Mark as mentioned to prevent inheritance
        # If no new location but this is a follow-up, inherit from last query
        elif not location_mentioned and is_follow_up and self.last_filters.get('location'):
            location = self.last_filters['location']

        if location:
            filtered_df = filtered_df[filtered_df['warehouse_location'] == location]
            filters_applied.append(f"Warehouse: {location}")
            current_filters['location'] = location

        # Category filter - check for new category in question
        category = None
        category_mentioned = False
        category_map = {
            'electronics': 'Electronics',
            'elektronik': 'Electronics',
            'books': 'Books',
            'kitap': 'Books',
            'home': 'Home & Kitchen',
            'ev': 'Home & Kitchen',
            'kitchen': 'Home & Kitchen',
            'mutfak': 'Home & Kitchen',
            'clothing': 'Clothing, Shoes & Jewelry',
            'clothes': 'Clothing, Shoes & Jewelry',
            'fashion': 'Clothing, Shoes & Jewelry',
            'giyim': 'Clothing, Shoes & Jewelry',
            'moda': 'Clothing, Shoes & Jewelry',
            'shoes': 'Clothing, Shoes & Jewelry',
            'jewelry': 'Clothing, Shoes & Jewelry',
            'sports': 'Sports & Outdoors',
            'spor': 'Sports & Outdoors',
            'beauty': 'Beauty & Personal Care',
            'güzellik': 'Beauty & Personal Care',
            'toys': 'Toys & Games',
            'oyuncak': 'Toys & Games',
            'pet': 'Pet Supplies',
            'automotive': 'Automotive',
            'tools': 'Tools & Home Improvement',
            'office': 'Office Products',
            'health': 'Health & Household',
        }
        for key, cat in category_map.items():
            if key in question_lower:
                category = cat
                category_mentioned = True
                break

        # CRITICAL: If new category mentioned, DON'T inherit old category
        # Only inherit if NO category mentioned at all
        if not category_mentioned and is_follow_up and self.last_filters.get('category'):
            category = self.last_filters['category']

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]
            filters_applied.append(f"Category: {category}")
            current_filters['category'] = category
        
        # Product name search (earbuds, ipad, macbook, etc.)
        product_keywords = self._extract_product_keywords(question)

        # If no new product keywords but this is a follow-up, inherit from last query
        if not product_keywords and is_follow_up and self.last_filters.get('product_keywords'):
            product_keywords = self.last_filters['product_keywords']
        elif product_keywords and is_follow_up and self.last_filters.get('product_keywords'):
            # Merge new keywords with old ones (don't replace)
            product_keywords = list(set(product_keywords + self.last_filters['product_keywords']))

        if product_keywords:
            pattern = '|'.join(product_keywords)
            filtered_df = filtered_df[filtered_df['product_name'].str.lower().str.contains(pattern, na=False)]
            filters_applied.append(f"Product: {', '.join(product_keywords)}")
            current_filters['product_keywords'] = product_keywords
        
        # Rating filter
        rating_match = re.search(r'rating[\'"]?\s*[><=]+\s*([\d.]+)|rating\s+([\d.]+)\s*(?:üstü|üzeri|den fazla|dan fazla)', question_lower)
        if not rating_match:
            rating_match = re.search(r'([\d.]+)\s*(?:üstü|üzeri|den fazla|dan fazla|üzerinde).*rating|rating.*?([\d.]+)\s*(?:üstü|üzeri)', question_lower)
        if rating_match:
            rating_val = float(rating_match.group(1) or rating_match.group(2))
            filtered_df = filtered_df[filtered_df['rating'] > rating_val]
            filters_applied.append(f"Rating > {rating_val}")
        
        # Price filter
        price_match = re.search(r'(?:fiyat|price)[\'"]?\s*[><=]+\s*([\d.]+)|(?:fiyat|price)\s+([\d.]+)\s*(?:üstü|üzeri|dolar)', question_lower)
        if not price_match:
            price_match = re.search(r'([\d.]+)\s*(?:dolar|usd|\$)?\s*(?:üstü|üzeri|üzerinde).*(?:fiyat|price)|(?:fiyat|price).*?([\d.]+)\s*(?:üstü|üzeri)', question_lower)
        if price_match:
            price_val = float(price_match.group(1) or price_match.group(2))
            filtered_df = filtered_df[filtered_df['price'] > price_val]
            filters_applied.append(f"Price > ${price_val}")
        
        # Stock filter
        stock_match = re.search(r'stok\s*([\d]+)\s*(?:üstü|üzeri|den fazla)|stock.*?([\d]+)', question_lower)
        if stock_match:
            stock_val = int(stock_match.group(1) or stock_match.group(2))
            filtered_df = filtered_df[filtered_df['stock_quantity'] > stock_val]
            filters_applied.append(f"Stock > {stock_val}")
        
        # Extract limit (how many items requested)
        limit = None
        limit_match = re.search(r'(\d+)\s*(?:tane|adet|ürün|tanesini|adedini)', question_lower)
        if not limit_match:
            limit_match = re.search(r'(?:ilk|en iyi|top)\s*(\d+)', question_lower)
        if limit_match:
            limit = int(limit_match.group(1))
        
        # Detect sorting requirements
        # Best rating + cheapest price combo
        if ('en yüksek' in question_lower or 'en iyi' in question_lower or 'en çok' in question_lower) and 'rating' in question_lower:
            if 'uygun' in question_lower or 'ucuz' in question_lower or 'düşük' in question_lower:
                # High rating, low price - sort by rating desc, then price asc
                filtered_df = filtered_df.dropna(subset=['price', 'rating'])
                filtered_df = filtered_df.sort_values(['rating', 'price'], ascending=[False, True])
                sort_info = "En Yüksek Rating + En Uygun Fiyat"
            else:
                # Just high rating
                filtered_df = filtered_df.sort_values('rating', ascending=False)
                sort_info = "En Yüksek Rating"
        elif 'en ucuz' in question_lower or 'en uygun' in question_lower or ('fiyat' in question_lower and 'düşük' in question_lower):
            filtered_df = filtered_df.dropna(subset=['price'])
            filtered_df = filtered_df.sort_values('price', ascending=True)
            sort_info = "En Uygun Fiyat"
        elif 'en pahalı' in question_lower or ('fiyat' in question_lower and 'yüksek' in question_lower):
            filtered_df = filtered_df.dropna(subset=['price'])
            filtered_df = filtered_df.sort_values('price', ascending=False)
            sort_info = "En Pahalı"
        elif 'en çok satan' in question_lower or 'satış' in question_lower:
            filtered_df = filtered_df.sort_values('daily_sales_avg', ascending=False)
            sort_info = "En Çok Satan"
        elif 'en çok stok' in question_lower or 'stok' in question_lower and 'en' in question_lower:
            filtered_df = filtered_df.sort_values('stock_quantity', ascending=False)
            sort_info = "En Çok Stok"
        
        # Determine query type and generate response
        count = len(filtered_df)

        # Store current filters for follow-up questions
        self.last_filters = current_filters

        # Store for follow-up
        filter_desc = " + ".join(filters_applied) if filters_applied else "Tüm ürünler"
        if sort_info:
            filter_desc = f"{filter_desc} ({sort_info})"
        self.last_query_result = {'df': filtered_df, 'description': filter_desc}

        # Handle "listele" / "göster" command - use last result if no new filters
        is_list_command = any(cmd in question_lower for cmd in ['listele', 'listede', 'liste', 'göster', 'bunları göster', 'bunları listele'])
        if is_list_command and not filters_applied and self.last_query_result is not None:
            # Use the previous query's results
            return self._format_product_list(
                self.last_query_result['df'],
                self.last_query_result['description'],
                limit,
                sort_info
            )

        # Detect "ne kadar" ambiguity - prefer count over sum for product queries
        is_quantity_question = 'ne kadar' in question_lower or 'kaç tane' in question_lower

        # Count query (only count, no list)
        if ('kaç' in question_lower or 'count' in question_lower or 'how many' in question_lower or 'sayı' in question_lower or is_quantity_question) and not ('liste' in question_lower or 'göster' in question_lower or 'list' in question_lower or 'show' in question_lower):
            self.last_query_type = 'count'
            if filters_applied:
                return f"**{count} items** ({filter_desc})"
            else:
                return f"**{count} items** found."

        # List query - with optional limit and sorting
        if 'liste' in question_lower or 'göster' in question_lower or 'list' in question_lower or 'show' in question_lower or 'which' in question_lower or 'hangi' in question_lower or limit:
            self.last_query_type = 'list'
            return self._format_product_list(filtered_df, filter_desc, limit, sort_info)

        # If filters were applied, return count
        if filters_applied:
            self.last_query_type = 'count'
            return f"**{count} items** ({filter_desc})"

        return None
    
    def _format_product_list(self, df: pd.DataFrame, description: str, limit: int = None, sort_info: str = None) -> str:
        """Format a DataFrame as a product list with optional limit."""
        if len(df) == 0:
            return "No products found matching these criteria."
        
        # Apply limit
        show_limit = limit if limit else 10
        display_df = df.head(show_limit)
        
        # Build response in ENGLISH
        if limit:
            title = f"Top {limit} Products ({description}):"
        else:
            title = f"{description} - {len(df)} items:"
        
        lines = [title, ""]
        
        for i, (_, row) in enumerate(display_df.iterrows(), 1):
            name = row['product_name'][:48] + "..." if len(row['product_name']) > 48 else row['product_name']
            price = f"${row['price']:.2f}" if pd.notna(row['price']) else "N/A"
            stock = int(row['stock_quantity'])
            rating = f"{row['rating']:.1f}" if pd.notna(row['rating']) else "N/A"
            # Compact: 1 line per product
            lines.append(f"{i}. {name} | {price} | Stock:{stock} | {rating}")
        
        if len(df) > show_limit and not limit:
            lines.append(f"\n... and {len(df) - show_limit} more items.")
        
        return "\n".join(lines)
    
    def _extract_product_keywords(self, question: str) -> list:
        """Extract product-related keywords from question."""
        question_lower = question.lower()
        
        # Common product keywords to look for
        product_terms = [
            'earbuds', 'earbud', 'kulaklık', 'headphones', 'airpods',
            'ipad', 'tablet', 'macbook', 'laptop',
            'iphone', 'phone', 'telefon',
            'watch', 'saat', 'apple watch',
            'tv', 'television', 'fire tv', 'echo',
            'kindle', 'e-reader',
            'bluetooth', 'wireless',
            'charger', 'şarj', 'cable', 'kablo',
            'speaker', 'hoparlör',
            'camera', 'kamera',
            'keyboard', 'klavye', 'mouse',
        ]
        
        found = []
        for term in product_terms:
            if term in question_lower:
                found.append(term)
        
        return found
    
    def _smart_data_query(self, question: str) -> str:
        """Use LLM to generate and execute pandas code for the question."""
        
        # Get column info for the LLM
        columns_info = """
DataFrame 'df' columns:
- asin: Product ID (string)
- product_name: Product name (string)
- category: Category (Electronics, Books, Home & Kitchen, etc.)
- price: Price in USD (float)
- rating: Rating 0-5 (float)
- review_count: Number of reviews (float)
- stock_quantity: Current stock (int)
- warehouse_location: Melbourne, Sydney, Brisbane, Perth (string)
- daily_sales_avg: Average daily sales units (float)
- minimum_stock_threshold: Min stock level (int)
- satisfaction_level: Excellent, Good, Average, Poor (string)
"""
        
        # Ask LLM to generate pandas code
        code_prompt = f"""Sen bir veri analisti asistanısın. Kullanıcının sorusuna göre pandas kodu üret.

{columns_info}

Kullanıcı Sorusu: {question}

Kurallar:
1. Sadece pandas kodu üret, açıklama yazma
2. DataFrame adı: df
3. Sonucu 'result' değişkenine ata
4. result bir string olmalı (kullanıcıya gösterilecek cevap)
5. Sayıları formatla (virgül, $ işareti vb.)
6. Türkçe cevap üret
7. Eğer liste gerekiyorsa en fazla 10 ürün göster

Örnek çıktı formatları:
- Sayı sorusu: result = f"Melbourne'da en çok stoklu ürün sayısı: {{count}} adet"
- Liste sorusu: result = "En çok stoklu ürünler:\\n" + "\\n".join([f"{{i+1}}. {{row['product_name'][:50]}} - Stok: {{row['stock_quantity']}}" for i, row in top_df.iterrows()])

Sadece Python kodu yaz:
```python
"""
        
        code_response = self.llm.invoke(code_prompt)
        code = code_response.content
        
        # Clean up the code
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Execute the code safely
        local_vars = {'df': self.df.copy(), 'pd': pd}
        exec(code, {"__builtins__": {}}, local_vars)
        
        result = local_vars.get('result', None)
        
        if result:
            return str(result)
        
        return None
    
    def _handle_summary_query(self, question: str) -> str:
        """Handle summary/statistics queries with calculations."""
        filtered_df = self.df.copy()
        location = None
        category = None
        
        # Find location filter
        for loc in ['melbourne', 'sydney', 'brisbane', 'perth']:
            if loc in question:
                location = loc.title()
                filtered_df = filtered_df[filtered_df['warehouse_location'] == location]
                break
        
        # Find category filter
        for cat in self.df['category'].unique():
            if cat.lower() in question:
                category = cat
                filtered_df = filtered_df[filtered_df['category'] == cat]
                break
        
        # Calculate statistics
        total_products = len(filtered_df)
        total_stock = filtered_df['stock_quantity'].sum()
        total_daily_sales = filtered_df['daily_sales_avg'].sum()
        
        # Calculate daily revenue (price * daily_sales_avg)
        filtered_df['daily_revenue'] = filtered_df['price'].fillna(0) * filtered_df['daily_sales_avg'].fillna(0)
        total_daily_revenue = filtered_df['daily_revenue'].sum()
        
        avg_price = filtered_df['price'].mean()
        avg_rating = filtered_df['rating'].mean()
        avg_stock = filtered_df['stock_quantity'].mean()
        
        # Low stock count
        low_stock_count = len(filtered_df[filtered_df['stock_quantity'] < filtered_df['minimum_stock_threshold']])
        
        # Build title
        title_parts = []
        if location:
            title_parts.append(f"{location} Deposu")
        if category:
            title_parts.append(f"{category}")
        title = " - ".join(title_parts) if title_parts else "Tüm Depolar"
        
        result = f"""## 📊 {title} Satış Özeti
### {title} Sales Summary

---

| Metrik | Değer |
|--------|-------|
| **Toplam Ürün (Total Products)** | {total_products:,} |
| **Toplam Stok (Total Stock)** | {total_stock:,} adet |
| **Günlük Satış (Daily Units Sold)** | {total_daily_sales:,.0f} adet |
| **Günlük Gelir (Daily Revenue)** | ${total_daily_revenue:,.2f} |
| **Ortalama Fiyat (Avg Price)** | ${avg_price:.2f} |
| **Ortalama Rating (Avg Rating)** | {avg_rating:.2f}/5 ⭐ |
| **Ortalama Stok (Avg Stock)** | {avg_stock:.0f} adet |
| **Düşük Stok Ürün Sayısı (Low Stock)** | {low_stock_count} ürün ⚠️ |

---

### 📈 Detaylı Hesaplamalar:
- **Günlük Satış**: Tüm ürünlerin `daily_sales_avg` toplamı
- **Günlük Gelir**: Her ürün için `price × daily_sales_avg` toplamı
"""
        
        # Add top sellers if asking about sales
        if 'satış' in question or 'sales' in question or 'gelir' in question:
            top_sellers = filtered_df.nlargest(5, 'daily_sales_avg')
            result += "\n### 🏆 En Çok Satan 5 Ürün:\n"
            for i, (_, row) in enumerate(top_sellers.iterrows(), 1):
                daily_rev = row['price'] * row['daily_sales_avg'] if pd.notna(row['price']) else 0
                result += f"{i}. {row['product_name'][:50]}...\n"
                result += f"   - Günlük Satış: {row['daily_sales_avg']:.0f} adet | Günlük Gelir: ${daily_rev:,.2f}\n"
        
        return result
    
    def _handle_average_query(self, question: str) -> str:
        """Handle average/mean queries."""
        results = []
        
        # Detect category
        categories = self.df['category'].unique()
        target_category = None
        for cat in categories:
            if cat.lower() in question or any(word in question for word in cat.lower().split()):
                target_category = cat
                break
        
        if target_category:
            cat_df = self.df[self.df['category'] == target_category]
        else:
            cat_df = self.df
        
        # Calculate averages
        if 'rating' in question or 'puan' in question or 'yıldız' in question:
            avg_rating = cat_df['rating'].mean()
            results.append(f"Ortalama Rating: {avg_rating:.2f}/5 yıldız")
        
        if 'fiyat' in question or 'price' in question:
            avg_price = cat_df['price'].mean()
            results.append(f"Ortalama Fiyat: ${avg_price:.2f}")
        
        if 'stok' in question or 'stock' in question:
            avg_stock = cat_df['stock_quantity'].mean()
            results.append(f"Ortalama Stok: {avg_stock:.0f} adet")
        
        if not results:
            # Default: show all averages
            avg_rating = cat_df['rating'].mean()
            avg_price = cat_df['price'].mean()
            avg_stock = cat_df['stock_quantity'].mean()
            category_name = target_category if target_category else "Tüm Kategoriler"
            results = [
                f"**{category_name} İstatistikleri:**",
                f"- Toplam Ürün: {len(cat_df)}",
                f"- Ortalama Rating: {avg_rating:.2f}/5 yıldız",
                f"- Ortalama Fiyat: ${avg_price:.2f}",
                f"- Ortalama Stok: {avg_stock:.0f} adet",
                f"- Toplam Stok: {cat_df['stock_quantity'].sum():,} adet"
            ]
        
        return "\n".join(results)
    
    def _handle_count_query(self, question: str) -> str:
        """Handle count queries with filters."""
        filtered_df = self.df.copy()
        filters_applied = []
        
        # Location filter
        for loc in ['melbourne', 'sydney', 'brisbane', 'perth']:
            if loc in question:
                filtered_df = filtered_df[filtered_df['warehouse_location'].str.lower() == loc]
                filters_applied.append(f"Warehouse: {loc.title()}")
                break
        
        # Category filter
        for cat in self.df['category'].unique():
            if cat.lower() in question or 'electronics' in question and 'Electronics' in cat:
                filtered_df = filtered_df[filtered_df['category'] == cat]
                filters_applied.append(f"Category: {cat}")
                break
        
        # Price filters
        price_match = re.search(r'(?:price|fiyat)[ıi]?\s*[><=]*\s*(\d+(?:\.\d+)?)', question)
        if price_match:
            price_val = float(price_match.group(1))
            if 'üzeri' in question or 'üstü' in question or '>' in question:
                filtered_df = filtered_df[filtered_df['price'] > price_val]
                filters_applied.append(f"Price > ${price_val}")
            elif 'altı' in question or 'altında' in question or '<' in question:
                filtered_df = filtered_df[filtered_df['price'] < price_val]
                filters_applied.append(f"Fiyat < ${price_val}")
        
        # Stock filters
        stock_match = re.search(r'(?:stock|stok)[_\s]*(?:quantity)?[ıi]?\s*[><=]*\s*(\d+)', question)
        if stock_match:
            stock_val = int(stock_match.group(1))
            if 'üzeri' in question or 'üstü' in question or '>' in question:
                filtered_df = filtered_df[filtered_df['stock_quantity'] > stock_val]
                filters_applied.append(f"Stock > {stock_val}")
            elif 'altı' in question or 'altında' in question or '<' in question:
                filtered_df = filtered_df[filtered_df['stock_quantity'] < stock_val]
                filters_applied.append(f"Stok < {stock_val}")
        
        count = len(filtered_df)
        
        result_parts = [f"**Sonuç: {count} ürün bulundu**"]
        if filters_applied:
            result_parts.append(f"Uygulanan filtreler: {', '.join(filters_applied)}")
        
        if count > 0 and count <= 10:
            result_parts.append("\n**Ürünler:**")
            for _, row in filtered_df.head(10).iterrows():
                result_parts.append(f"- {row['product_name'][:50]}... | Stok: {row['stock_quantity']} | Fiyat: ${row['price']:.2f}")
        elif count > 10:
            result_parts.append(f"\n(İlk 10 ürün gösteriliyor)")
            for _, row in filtered_df.head(10).iterrows():
                result_parts.append(f"- {row['product_name'][:50]}... | Stok: {row['stock_quantity']} | Fiyat: ${row['price']:.2f}")
        
        return "\n".join(result_parts)
    
    def _handle_location_query(self, question: str) -> str:
        """Handle location-based queries."""
        filtered_df = self.df.copy()
        location = None
        filters_applied = []
        
        # Find location
        for loc in ['melbourne', 'sydney', 'brisbane', 'perth']:
            if loc in question:
                location = loc.title()
                filtered_df = filtered_df[filtered_df['warehouse_location'] == location]
                filters_applied.append(f"Warehouse: {location}")
                break
        
        # Category filter
        if 'electronics' in question:
            filtered_df = filtered_df[filtered_df['category'] == 'Electronics']
            filters_applied.append("Category: Electronics")
        elif 'books' in question or 'kitap' in question:
            filtered_df = filtered_df[filtered_df['category'] == 'Books']
            filters_applied.append("Category: Books")
        
        # Extract numeric conditions
        numbers = re.findall(r'(\d+(?:\.\d+)?)', question)
        
        # Apply stock filter
        if 'stock' in question or 'stok' in question:
            for num in numbers:
                num_val = float(num)
                if num_val > 10:  # Likely a stock threshold
                    if 'üzeri' in question or 'üstü' in question:
                        filtered_df = filtered_df[filtered_df['stock_quantity'] > num_val]
                        filters_applied.append(f"Stock > {int(num_val)}")
                        break
        
        # Apply price filter
        if 'price' in question or 'fiyat' in question:
            for num in numbers:
                num_val = float(num)
                if num_val > 10:  # Likely a price threshold
                    if 'üzeri' in question or 'üstü' in question:
                        filtered_df = filtered_df[filtered_df['price'] > num_val]
                        filters_applied.append(f"Price > ${num_val}")
                    elif 'altı' in question or 'altında' in question:
                        filtered_df = filtered_df[filtered_df['price'] < num_val]
                        filters_applied.append(f"Fiyat < ${num_val}")
        
        count = len(filtered_df)
        
        result_parts = [f"**Sonuç: {count} ürün bulundu**"]
        if filters_applied:
            result_parts.append(f"Filtreler: {', '.join(filters_applied)}")
        
        if count > 0:
            result_parts.append("\n**Ürünler:**")
            for _, row in filtered_df.head(15).iterrows():
                result_parts.append(f"- {row['product_name'][:60]}...")
                result_parts.append(f"  Stok: {row['stock_quantity']} | Fiyat: ${row['price']:.2f} | Rating: {row['rating']}")
        
        return "\n".join(result_parts)
    
    def _handle_top_query(self, question: str) -> str:
        """Handle top/best queries."""
        filtered_df = self.df.copy()
        
        # Category filter
        for cat in self.df['category'].unique():
            if cat.lower() in question:
                filtered_df = filtered_df[filtered_df['category'] == cat]
                break
        
        # Determine sort column
        if 'rating' in question or 'puan' in question or 'yıldız' in question:
            sorted_df = filtered_df.nlargest(10, 'rating')
            metric = "Rating"
        elif 'fiyat' in question or 'price' in question:
            if 'ucuz' in question or 'düşük' in question:
                sorted_df = filtered_df.nsmallest(10, 'price')
                metric = "En Düşük Fiyat"
            else:
                sorted_df = filtered_df.nlargest(10, 'price')
                metric = "En Yüksek Fiyat"
        elif 'stok' in question or 'stock' in question:
            sorted_df = filtered_df.nlargest(10, 'stock_quantity')
            metric = "Stok"
        elif 'review' in question or 'inceleme' in question or 'yorum' in question:
            sorted_df = filtered_df.nlargest(10, 'review_count')
            metric = "İnceleme Sayısı"
        else:
            sorted_df = filtered_df.nlargest(10, 'rating')
            metric = "Rating"
        
        result_parts = [f"**En Yüksek {metric} - Top 10:**\n"]
        
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            result_parts.append(f"{i}. **{row['product_name'][:60]}**")
            result_parts.append(f"   - Category: {row['category']}")
            result_parts.append(f"   - Rating: {row['rating']}/5 ({row['review_count']:.0f} inceleme)")
            result_parts.append(f"   - Fiyat: ${row['price']:.2f}")
            result_parts.append(f"   - Stok: {row['stock_quantity']} ({row['warehouse_location']})")
            result_parts.append("")
        
        return "\n".join(result_parts)
    
    def _handle_low_stock_query(self, question: str) -> str:
        """Handle low stock queries."""
        low_stock_df = self.df[self.df['stock_quantity'] < self.df['minimum_stock_threshold']]
        
        # Category filter
        for cat in self.df['category'].unique():
            if cat.lower() in question:
                low_stock_df = low_stock_df[low_stock_df['category'] == cat]
                break
        
        low_stock_df = low_stock_df.sort_values('stock_quantity')
        
        result_parts = [f"**⚠️ Düşük Stok Uyarısı: {len(low_stock_df)} ürün**\n"]
        
        for _, row in low_stock_df.head(20).iterrows():
            deficit = row['minimum_stock_threshold'] - row['stock_quantity']
            result_parts.append(f"- **{row['product_name'][:50]}**")
            result_parts.append(f"  Stok: {row['stock_quantity']} (Min: {row['minimum_stock_threshold']}, Eksik: {deficit})")
            result_parts.append(f"  Warehouse: {row['warehouse_location']} | Restock: {row['restock_date'][:10]}")
            result_parts.append("")
        
        return "\n".join(result_parts)
    
    def ask(self, question: str) -> str:
        """Ask a question - LLM-driven intent routing (NO MORE PATTERN MATCHING!)."""
        # Normalize common mis-hearings from speech recognition
        question = self._normalize_question(question)

        try:
            # Step 0: Check for transfer queries FIRST (before general intent)
            transfer_result = self._check_for_transfer_query(question)
            if transfer_result:
                return transfer_result

            # Step 1: LLM extracts intent and entities
            intent = self.intent_chain.invoke({"question": question})
            intent_type = intent.get('intent', 'unknown')

            # Step 2: Route based on intent
            if intent_type == 'recommendation':
                # Warehouse recommendation
                return self._get_smart_warehouse_recommendation(
                    category=intent.get('category'),
                    product_keywords=intent.get('product_keywords'),
                    exclude_location=intent.get('exclude_locations', [None])[0] if intent.get('exclude_locations') else None
                )

            elif intent_type == 'simulation':
                # What-if scenarios
                return self._simulate_scenario(intent)

            elif intent_type == 'count':
                # Count/total queries
                # BUT: If user says "show me", "list", "display" → they want a LIST not count
                question_lower = question.lower()
                wants_list = any(word in question_lower for word in ['show', 'list', 'display', 'give me'])
                
                locations = intent.get('locations', [])
                category = intent.get('category')

                filtered_df = self.df.copy()
                if category:
                    filtered_df = filtered_df[filtered_df['category'] == category]
                if locations:
                    filtered_df = filtered_df[filtered_df['warehouse_location'].isin(locations)]

                # Check if user specified a limit (e.g., "show me 3 books")
                limit_match = re.search(r'(\d+)', question_lower)
                limit = int(limit_match.group(1)) if limit_match else None
                
                if wants_list or limit:
                    # User wants to see products, not just count
                    desc_parts = []
                    if category:
                        desc_parts.append(f"Category: {category}")
                    if locations:
                        desc_parts.append(f"Locations: {', '.join(locations)}")
                    desc = " + ".join(desc_parts) if desc_parts else "All items"
                    return self._format_product_list(filtered_df, desc, limit)
                else:
                    # Just count
                    count = len(filtered_df)
                    desc_parts = []
                    if category:
                        desc_parts.append(f"Category: {category}")
                    if locations:
                        desc_parts.append(f"Locations: {', '.join(locations)}")
                    desc = " + ".join(desc_parts) if desc_parts else "All items"
                    return f"**{count} items** ({desc})"

            elif intent_type == 'comparison':
                # Comparison queries
                locations = intent.get('locations', [])
                if len(locations) >= 2:
                    return self._breakdown_by_locations(question, locations)
                else:
                    return self._breakdown_by_locations(question)

            elif intent_type == 'product_query' or intent_type == 'unknown':
                # Fall back to RAG for product-specific or unknown queries
                context_note = ""
                if self.last_filters.get('location'):
                    context_note += f" [Context: Last query was about {self.last_filters['location']} warehouse]"
                if self.last_filters.get('category'):
                    context_note += f" [Context: Last query was about {self.last_filters['category']} category]"

                enhanced_question = question + context_note if context_note else question
                return self.rag_chain.invoke(enhanced_question)

            else:
                return "⚠️ I couldn't understand your question. Please try rephrasing."

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            # Fall back to RAG on error
            try:
                return self.rag_chain.invoke(question)
            except:
                return "I encountered an error. Please try rephrasing your question."

    def _normalize_question(self, question: str) -> str:
        """Fix frequent ASR mis-hearings for locations/categories."""
        q = question.lower()
        replacements = {
            # Perth phonetic variants (TR) - Whisper often mishears as kört, pört, port
            "kört": "perth",
            "körte": "perth",
            "körtte": "perth",
            "pört": "perth",
            "pörth": "perth",
            "pörtte": "perth",
            "pörtteki": "perth",
            "pörte": "perth",
            "port şehrinde": "perth",
            "port'ta": "perth",
            "portta": "perth",
            # Sydney phonetic variants
            "sıdney": "sydney",
            "sıdni": "sydney",
            "sidni": "sydney",
            "sidneyde": "sydney",
            "sydney'de": "sydney",  # Keep location suffix
            "siz neydi": "sydney",  # Common Whisper mistake
            "siz ney": "sydney",
            "sidney": "sydney",
            "sidneydi": "sydney",
            # Melbourne phonetic variants
            "melbourn": "melbourne",
            "melbırn": "melbourne",
            "melbörn": "melbourne",
            "melvin": "melbourne",  # Common mis-hearing
            "malbun": "melbourne",
            "malbın": "melbourne",
            "melbun": "melbourne",
            "melbourne'da": "melbourne",  # Keep location suffix
            "melborn": "melbourne",
            "melbörün": "melbourne",
            # Brisbane phonetic variants
            "brisbeyn": "brisbane",
            "brizbeyn": "brisbane",
            "brisbane'de": "brisbane",  # Keep location suffix
            "brisbın": "brisbane",
            "brisbane": "brisbane",
        }
        for bad, good in replacements.items():
            q = q.replace(bad, good)
        # Preserve original casing minimally
        return q
    
    def run_interactive(self):
        """Run interactive chat session."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]🤖 AI Operations Assistant v2.0[/bold green]\n"
            "[dim]E-Commerce Inventory Chatbot (Hybrid RAG + Data)[/dim]\n\n"
            "Örnek sorular:\n"
            "• Fire TV Stick'ten kaç tane stokta var?\n"
            "• En yüksek rating'li Electronics ürünü hangisi?\n"
            "• Melbourne'da stock 150 üzeri kaç ürün var?\n"
            "• Books kategorisinin ortalama rating'i nedir?\n"
            "• Low stock alert olan ürünler hangileri?\n\n"
            "[yellow]Çıkmak için 'q' veya 'exit' yazın[/yellow]",
            title="🛒 Hoş Geldiniz",
            border_style="green"
        ))
        
        while True:
            try:
                self.console.print()
                question = Prompt.ask("[bold cyan]Soru[/bold cyan]")
                
                if question.lower() in ['q', 'exit', 'quit', 'çık', 'çıkış']:
                    self.console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                # Show thinking indicator
                with self.console.status("[bold green]Düşünüyorum...[/bold green]"):
                    answer = self.ask(question)
                
                # Display answer
                self.console.print()
                self.console.print(Panel(
                    Markdown(answer),
                    title="[bold green]🤖 Cevap[/bold green]",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]👋 Görüşmek üzere![/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]❌ Hata: {e}[/red]")


def rebuild_vectorstore():
    """Rebuild the vector store from scratch."""
    console = Console()
    console.print("[yellow]🔄 Rebuilding vector store...[/yellow]")
    
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        console.print("[green]✓ Old vector store removed[/green]")
    
    chatbot = ProductChatbot()
    console.print("[green]✓ Vector store rebuilt successfully![/green]")
    return chatbot


def main():
    """Main entry point."""
    import sys
    
    console = Console()
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   AI Operations Assistant v2.0           ║[/bold blue]")
    console.print("[bold blue]║   Hybrid RAG + Data Analytics            ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════════╝[/bold blue]")
    console.print()
    
    if "--rebuild" in sys.argv:
        chatbot = rebuild_vectorstore()
    else:
        chatbot = ProductChatbot()
    
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
