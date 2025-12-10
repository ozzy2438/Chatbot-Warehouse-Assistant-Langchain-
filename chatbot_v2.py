"""
AI Operations Assistant v2.0 - E-Commerce Inventory Chatbot
============================================================
Production-ready chatbot using OpenAI Function Calling for robust intent handling.

Key Improvements over v1:
- OpenAI Function Calling instead of brittle pattern matching
- Clean separation of tools and execution
- Comprehensive test coverage
- Professional error handling

Tech Stack:
- LLM: OpenAI GPT-4o-mini with Function Calling
- Vector DB: ChromaDB (for product search)
- Data: Pandas DataFrame for analytics
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from openai import OpenAI

# LangChain imports (for RAG only)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Rich for terminal output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

load_dotenv()

# Constants
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "products"

# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_warehouse_recommendation",
            "description": "Find the best warehouse/city for a product category. Use when user asks 'where should I go', 'which warehouse', 'best location for X', or mentions a location being out of stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Product category to find warehouse for"
                    },
                    "exclude_location": {
                        "type": "string",
                        "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"],
                        "description": "Location to exclude (e.g., if out of stock or full)"
                    },
                    "product_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific product keywords to filter by (e.g., ['airpods', 'earbuds'])"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_items",
            "description": "Count the number of products matching criteria. Use for 'how many', 'count', 'total items' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Filter by category"
                    },
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"]
                        },
                        "description": "Filter by warehouse locations"
                    },
                    "product_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by product name keywords"
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum rating filter (e.g., 4.5)"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter in USD"
                    },
                    "min_stock": {
                        "type": "integer",
                        "description": "Minimum stock quantity filter"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_products",
            "description": "List products matching criteria with sorting. Use for 'show me', 'list', 'which products', 'top X' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Filter by category"
                    },
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"]
                        },
                        "description": "Filter by warehouse locations"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "rating_desc", "stock_desc", "sales_desc"],
                        "description": "Sort order: price_asc (cheapest), price_desc (expensive), rating_desc (best rated), stock_desc (most stock), sales_desc (best selling)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of products to return (default: 10)"
                    },
                    "product_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by product name keywords"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_locations",
            "description": "Compare stock levels between warehouses. Use for 'compare', 'difference between', 'breakdown by location' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"]
                        },
                        "description": "Locations to compare (2 or more)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Filter comparison by category"
                    }
                },
                "required": ["locations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_inventory_change",
            "description": "Simulate adding or removing inventory. Use for 'what if', 'if I order', 'if I add/remove X items' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract"],
                        "description": "Whether to add or subtract items"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to add/subtract"
                    },
                    "location": {
                        "type": "string",
                        "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"],
                        "description": "Target warehouse location"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Product category"
                    }
                },
                "required": ["operation", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for specific products by name or description using semantic search. Use for product-specific questions like 'tell me about AirPods', 'what alternatives to X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name, description, or question about products)"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_stock_alerts",
            "description": "Get products that are below minimum stock threshold. Use for 'low stock', 'need restock', 'stock alerts' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["Sydney", "Melbourne", "Brisbane", "Perth"],
                        "description": "Filter by warehouse location"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Electronics", "Books", "Home & Kitchen", "Clothing, Shoes & Jewelry"],
                        "description": "Filter by category"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of alerts to show (default: 10)"
                    }
                },
                "required": []
            }
        }
    }
]


class ProductChatbot:
    """AI-powered chatbot using OpenAI Function Calling for robust intent handling."""

    def __init__(self):
        self.console = Console()
        self._check_api_key()
        self._load_dataframe()
        self._init_openai_client()
        self._init_vectorstore()
        
        # Conversation history for multi-turn
        self.conversation_history = []
        self.last_context = {}

    def _check_api_key(self):
        """Check if OpenAI API key is set."""
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("[bold red]OPENAI_API_KEY not found![/bold red]")
            raise SystemExit(1)

    def _load_dataframe(self):
        """Load product data into DataFrame."""
        self.console.print("[cyan]Loading product database...[/cyan]")
        csv_path = DATA_DIR / "products_for_ai.csv"
        self.df = pd.read_csv(csv_path)
        
        # Load alternatives if available
        alt_path = DATA_DIR / "products_with_alternatives.csv"
        if alt_path.exists():
            df_alt = pd.read_csv(alt_path)
            self.df = self.df.merge(
                df_alt[['asin', 'similar_price_alternatives', 'higher_rated_alternatives']],
                on='asin', how='left'
            )
        
        self.console.print(f"[green]Loaded {len(self.df)} products[/green]")

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        self.client = OpenAI()
        self.console.print("[green]OpenAI client initialized[/green]")

    def _init_vectorstore(self):
        """Initialize ChromaDB for semantic search."""
        self.console.print("[cyan]Setting up vector database...[/cyan]")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            documents = self._create_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.console.print("[green]Vector store ready[/green]")

    def _create_documents(self) -> list[Document]:
        """Create documents for vector store."""
        documents = []
        for _, row in self.df.iterrows():
            content = f"""Product: {row['product_name']}
ASIN: {row['asin']}
Category: {row['category']}
Price: ${row['price']:.2f} if pd.notna(row['price']) else 'N/A'
Rating: {row['rating']}/5 ({row['review_count']:.0f} reviews)
Stock: {row['stock_quantity']} units in {row['warehouse_location']}
Satisfaction: {row['satisfaction_level']}"""
            
            metadata = {
                "asin": str(row['asin']),
                "category": str(row['category']),
                "warehouse_location": str(row['warehouse_location']),
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    # =====================
    # TOOL IMPLEMENTATIONS
    # =====================

    def tool_get_warehouse_recommendation(
        self,
        category: Optional[str] = None,
        exclude_location: Optional[str] = None,
        product_keywords: Optional[list] = None
    ) -> str:
        """Find best warehouse based on stock levels."""
        filtered_df = self.df.copy()

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        if product_keywords:
            pattern = '|'.join(product_keywords)
            filtered_df = filtered_df[
                filtered_df['product_name'].str.lower().str.contains(pattern, na=False)
            ]

        if exclude_location:
            filtered_df = filtered_df[filtered_df['warehouse_location'] != exclude_location]

        if len(filtered_df) == 0:
            return "No stock found matching your criteria."

        # Aggregate by warehouse
        warehouse_stats = filtered_df.groupby('warehouse_location').agg({
            'stock_quantity': 'sum',
            'asin': 'count'
        }).sort_values('stock_quantity', ascending=False)

        if len(warehouse_stats) == 0:
            return "No alternative warehouses available."

        lines = []
        
        # Primary recommendation
        best = warehouse_stats.index[0]
        best_stock = int(warehouse_stats.iloc[0]['stock_quantity'])
        best_items = int(warehouse_stats.iloc[0]['asin'])

        lines.append(f"**Primary Recommendation: {best}**")
        lines.append(f"   - Stock Available: **{best_stock} units** ({best_items} different items)")
        lines.append(f"   - Reason: Highest inventory for this category")

        # Alternatives
        if len(warehouse_stats) > 1:
            lines.append("\n**Alternative Options:**")
            for i in range(1, min(len(warehouse_stats), 3)):
                alt = warehouse_stats.index[i]
                alt_stock = int(warehouse_stats.iloc[i]['stock_quantity'])
                alt_items = int(warehouse_stats.iloc[i]['asin'])
                lines.append(f"   {i}. {alt}: {alt_stock} units ({alt_items} items)")

        # Context about excluded location
        if exclude_location:
            excluded_stock = self.df[
                (self.df['warehouse_location'] == exclude_location) &
                (self.df['category'] == category if category else True)
            ]['stock_quantity'].sum()

            if excluded_stock == 0:
                lines.append(f"\n**Note:** {exclude_location} is OUT OF STOCK for this category")
            else:
                lines.append(f"\n**Note:** {exclude_location} excluded as requested ({int(excluded_stock)} units there)")

        return "\n".join(lines)

    def tool_count_items(
        self,
        category: Optional[str] = None,
        locations: Optional[list] = None,
        product_keywords: Optional[list] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None,
        min_stock: Optional[int] = None
    ) -> str:
        """Count products matching criteria."""
        filtered_df = self.df.copy()
        filters_applied = []

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]
            filters_applied.append(f"Category: {category}")

        if locations:
            filtered_df = filtered_df[filtered_df['warehouse_location'].isin(locations)]
            filters_applied.append(f"Locations: {', '.join(locations)}")

        if product_keywords:
            pattern = '|'.join(product_keywords)
            filtered_df = filtered_df[
                filtered_df['product_name'].str.lower().str.contains(pattern, na=False)
            ]
            filters_applied.append(f"Keywords: {', '.join(product_keywords)}")

        if min_rating:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
            filters_applied.append(f"Rating >= {min_rating}")

        if max_price:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
            filters_applied.append(f"Price <= ${max_price}")

        if min_stock:
            filtered_df = filtered_df[filtered_df['stock_quantity'] >= min_stock]
            filters_applied.append(f"Stock >= {min_stock}")

        count = len(filtered_df)
        total_stock = int(filtered_df['stock_quantity'].sum())

        result = f"**{count} products** found"
        if filters_applied:
            result += f" ({', '.join(filters_applied)})"
        result += f"\n**Total stock:** {total_stock:,} units"

        return result

    def tool_list_products(
        self,
        category: Optional[str] = None,
        locations: Optional[list] = None,
        sort_by: Optional[str] = None,
        limit: int = 10,
        product_keywords: Optional[list] = None
    ) -> str:
        """List products with optional sorting."""
        filtered_df = self.df.copy()
        filters_applied = []

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]
            filters_applied.append(f"Category: {category}")

        if locations:
            filtered_df = filtered_df[filtered_df['warehouse_location'].isin(locations)]
            filters_applied.append(f"Locations: {', '.join(locations)}")

        if product_keywords:
            pattern = '|'.join(product_keywords)
            filtered_df = filtered_df[
                filtered_df['product_name'].str.lower().str.contains(pattern, na=False)
            ]
            filters_applied.append(f"Keywords: {', '.join(product_keywords)}")

        # Apply sorting
        sort_map = {
            'price_asc': ('price', True),
            'price_desc': ('price', False),
            'rating_desc': ('rating', False),
            'stock_desc': ('stock_quantity', False),
            'sales_desc': ('daily_sales_avg', False),
        }

        sort_label = ""
        if sort_by and sort_by in sort_map:
            col, asc = sort_map[sort_by]
            filtered_df = filtered_df.dropna(subset=[col])
            filtered_df = filtered_df.sort_values(col, ascending=asc)
            sort_label = f" (sorted by {sort_by.replace('_', ' ')})"

        if len(filtered_df) == 0:
            return "No products found matching your criteria."

        # Build response
        display_df = filtered_df.head(limit)
        
        title = f"**Top {len(display_df)} Products{sort_label}**"
        if filters_applied:
            title += f"\nFilters: {', '.join(filters_applied)}"

        lines = [title, ""]
        for i, (_, row) in enumerate(display_df.iterrows(), 1):
            name = row['product_name'][:50] + "..." if len(row['product_name']) > 50 else row['product_name']
            price = f"${row['price']:.2f}" if pd.notna(row['price']) else "N/A"
            rating = f"{row['rating']:.1f}" if pd.notna(row['rating']) else "N/A"
            stock = int(row['stock_quantity'])
            lines.append(f"{i}. {name}")
            lines.append(f"   Price: {price} | Rating: {rating}/5 | Stock: {stock} | {row['warehouse_location']}")

        if len(filtered_df) > limit:
            lines.append(f"\n... and {len(filtered_df) - limit} more products")

        return "\n".join(lines)

    def tool_compare_locations(
        self,
        locations: list,
        category: Optional[str] = None
    ) -> str:
        """Compare stock between warehouses."""
        filtered_df = self.df.copy()

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        # Get stats for requested locations
        stats = filtered_df.groupby('warehouse_location').agg({
            'stock_quantity': 'sum',
            'asin': 'count',
            'price': 'mean',
            'rating': 'mean'
        }).round(2)

        lines = [f"**Warehouse Comparison**"]
        if category:
            lines.append(f"Category: {category}\n")
        else:
            lines.append("")

        total = 0
        for loc in locations:
            if loc in stats.index:
                stock = int(stats.loc[loc, 'stock_quantity'])
                items = int(stats.loc[loc, 'asin'])
                avg_price = stats.loc[loc, 'price']
                avg_rating = stats.loc[loc, 'rating']
                lines.append(f"**{loc}:**")
                lines.append(f"   - Total Stock: {stock:,} units")
                lines.append(f"   - Products: {items} items")
                lines.append(f"   - Avg Price: ${avg_price:.2f}")
                lines.append(f"   - Avg Rating: {avg_rating:.2f}/5")
                lines.append("")
                total += stock
            else:
                lines.append(f"**{loc}:** No data")

        lines.append(f"**Combined Total:** {total:,} units")

        return "\n".join(lines)

    def tool_simulate_inventory_change(
        self,
        operation: str,
        quantity: int,
        location: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """Simulate inventory changes."""
        filtered_df = self.df.copy()

        if location:
            filtered_df = filtered_df[filtered_df['warehouse_location'] == location]
        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        current_total = len(filtered_df)
        current_stock = int(filtered_df['stock_quantity'].sum())

        if operation == 'subtract':
            if quantity > current_stock:
                return f"**Cannot remove {quantity} units** - only {current_stock:,} units available!"
            
            new_stock = current_stock - quantity
            change_pct = (quantity / current_stock * 100) if current_stock > 0 else 0
            
            lines = [
                f"**Simulation: Remove {quantity} units**",
                "",
                f"- Current stock: {current_stock:,} units",
                f"- After removal: {new_stock:,} units",
                f"- Change: -{quantity} units ({-change_pct:.1f}%)",
            ]
        else:  # add
            new_stock = current_stock + quantity
            change_pct = (quantity / current_stock * 100) if current_stock > 0 else 100
            
            lines = [
                f"**Simulation: Add {quantity} units**",
                "",
                f"- Current stock: {current_stock:,} units",
                f"- After addition: {new_stock:,} units",
                f"- Change: +{quantity} units (+{change_pct:.1f}%)",
            ]

        if location:
            lines.append(f"\nLocation: {location}")
        if category:
            lines.append(f"Category: {category}")

        return "\n".join(lines)

    def tool_search_products(self, query: str, num_results: int = 5) -> str:
        """Search products using semantic search."""
        docs = self.retriever.invoke(query)[:num_results]
        
        if not docs:
            return "No products found matching your query."

        lines = [f"**Search Results for '{query}':**\n"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"**{i}.** {doc.page_content}\n")

        return "\n".join(lines)

    def tool_get_low_stock_alerts(
        self,
        location: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """Get products below minimum stock threshold."""
        filtered_df = self.df[
            self.df['stock_quantity'] < self.df['minimum_stock_threshold']
        ].copy()

        if location:
            filtered_df = filtered_df[filtered_df['warehouse_location'] == location]
        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        filtered_df = filtered_df.sort_values('stock_quantity')

        if len(filtered_df) == 0:
            return "No low stock alerts! All products are above minimum threshold."

        lines = [f"**Low Stock Alerts: {len(filtered_df)} products need attention**\n"]

        for _, row in filtered_df.head(limit).iterrows():
            deficit = int(row['minimum_stock_threshold'] - row['stock_quantity'])
            name = row['product_name'][:45] + "..." if len(row['product_name']) > 45 else row['product_name']
            lines.append(f"- **{name}**")
            lines.append(f"  Stock: {int(row['stock_quantity'])} (Min: {int(row['minimum_stock_threshold'])}, Need: +{deficit})")
            lines.append(f"  Location: {row['warehouse_location']} | Restock: {str(row['restock_date'])[:10]}")
            lines.append("")

        if len(filtered_df) > limit:
            lines.append(f"... and {len(filtered_df) - limit} more alerts")

        return "\n".join(lines)

    # =====================
    # MAIN ASK METHOD
    # =====================

    def ask(self, question: str) -> str:
        """Process question using OpenAI Function Calling."""
        try:
            # Build messages with system prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are an AI Factory Manager and inventory advisor for an e-commerce warehouse system.
You help users manage inventory across warehouses in Sydney, Melbourne, Brisbane, and Perth.
Product categories: Electronics, Books, Home & Kitchen, Clothing Shoes & Jewelry.

IMPORTANT RULES:
1. Use ONE tool per question - pick the most appropriate one
2. For 'where should I go/send' questions, use get_warehouse_recommendation
3. For 'how many' questions, use count_items
4. For 'show me', 'list', 'top X' questions, use list_products
5. For comparisons between locations, use compare_locations
6. For 'what if' scenarios, use simulate_inventory_change
7. For product-specific searches, use search_products
8. For stock alerts, use get_low_stock_alerts
9. Always respond in ENGLISH
10. Be concise and professional
11. IMPORTANT: Only call ONE tool at a time, not multiple"""
                },
                {"role": "user", "content": question}
            ]

            # Call OpenAI with tools
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1
            )

            message = response.choices[0].message

            # Check if tools were called
            if message.tool_calls:
                # Handle ALL tool calls (parallel function calling)
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    result = self._execute_tool(function_name, arguments)
                    
                    # Add tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # Get final response after all tools executed
                final_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1
                )

                return final_response.choices[0].message.content

            # No tool called - direct response
            return message.content or "I couldn't process your question. Please try rephrasing."

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return f"An error occurred: {str(e)}"

    def _execute_tool(self, function_name: str, arguments: dict) -> str:
        """Execute a tool function by name."""
        tool_map = {
            "get_warehouse_recommendation": self.tool_get_warehouse_recommendation,
            "count_items": self.tool_count_items,
            "list_products": self.tool_list_products,
            "compare_locations": self.tool_compare_locations,
            "simulate_inventory_change": self.tool_simulate_inventory_change,
            "search_products": self.tool_search_products,
            "get_low_stock_alerts": self.tool_get_low_stock_alerts,
        }

        if function_name in tool_map:
            return tool_map[function_name](**arguments)
        else:
            return f"Unknown tool: {function_name}"

    def run_interactive(self):
        """Run interactive chat session."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]AI Operations Assistant v2.0[/bold green]\n"
            "[dim]Function Calling Edition - Production Ready[/dim]\n\n"
            "Example questions:\n"
            "- Best warehouse for books if Brisbane is full?\n"
            "- How many electronics in Sydney?\n"
            "- Show me top 5 cheapest products\n"
            "- Compare Melbourne and Perth stock\n"
            "- What if I remove 100 items from Sydney?\n\n"
            "[yellow]Type 'q' or 'exit' to quit[/yellow]",
            title="Welcome",
            border_style="green"
        ))

        while True:
            try:
                self.console.print()
                question = Prompt.ask("[bold cyan]Question[/bold cyan]")

                if question.lower() in ['q', 'exit', 'quit']:
                    self.console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not question.strip():
                    continue

                with self.console.status("[bold green]Thinking...[/bold green]"):
                    answer = self.ask(question)

                self.console.print()
                self.console.print(Panel(
                    Markdown(answer),
                    title="[bold green]Answer[/bold green]",
                    border_style="green"
                ))

            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    console = Console()
    console.print()
    console.print("[bold blue]AI Operations Assistant v2.0[/bold blue]")
    console.print("[dim]OpenAI Function Calling Edition[/dim]")
    console.print()

    chatbot = ProductChatbot()
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
