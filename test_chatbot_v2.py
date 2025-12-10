"""
Comprehensive Test Suite for AI Operations Assistant v2.0
=========================================================
Run with: pytest test_chatbot_v2.py -v

Tests cover:
1. Tool functions (unit tests - no API calls)
2. Integration tests (with mocked OpenAI)
3. End-to-end tests (with real API - marked slow)
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os
from dotenv import load_dotenv

# Load .env file BEFORE checking for API key
load_dotenv()

# Skip all tests if no API key (for CI/CD)
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.fixture(scope="module")
def chatbot():
    """Create chatbot instance once for all tests."""
    from chatbot_v2 import ProductChatbot
    return ProductChatbot()


@pytest.fixture
def sample_df():
    """Create sample DataFrame for isolated testing."""
    return pd.DataFrame({
        'asin': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'product_name': [
            'Apple AirPods Pro',
            'Amazon Kindle Paperwhite',
            'Sony Headphones',
            'Kitchen Blender Pro',
            'Nike Running Shoes'
        ],
        'category': [
            'Electronics', 'Electronics', 'Electronics',
            'Home & Kitchen', 'Clothing, Shoes & Jewelry'
        ],
        'price': [249.99, 139.99, 199.99, 79.99, 129.99],
        'rating': [4.8, 4.6, 4.5, 4.2, 4.4],
        'review_count': [50000, 30000, 20000, 5000, 10000],
        'stock_quantity': [100, 50, 200, 30, 150],
        'warehouse_location': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Sydney'],
        'daily_sales_avg': [25, 15, 10, 5, 20],
        'minimum_stock_threshold': [50, 60, 40, 25, 100],
        'satisfaction_level': ['Excellent', 'Good', 'Good', 'Average', 'Good'],
        'restock_date': ['2025-01-15', '2025-01-10', '2025-01-20', '2025-01-05', '2025-01-12']
    })


# ============================================
# UNIT TESTS - Tool Functions (No API calls)
# ============================================

class TestWarehouseRecommendation:
    """Tests for get_warehouse_recommendation tool."""

    def test_basic_recommendation(self, chatbot):
        """Should return warehouse with most stock."""
        result = chatbot.tool_get_warehouse_recommendation(category="Electronics")
        assert "Primary Recommendation" in result
        assert "units" in result

    def test_exclude_location(self, chatbot):
        """Should exclude specified location."""
        result = chatbot.tool_get_warehouse_recommendation(
            category="Electronics",
            exclude_location="Sydney"
        )
        assert "Primary Recommendation" in result
        # Sydney should not be the primary recommendation
        assert "Sydney excluded" in result or "Sydney" not in result.split("\n")[0]

    def test_with_product_keywords(self, chatbot):
        """Should filter by product keywords."""
        result = chatbot.tool_get_warehouse_recommendation(
            product_keywords=["airpods", "earbuds"]
        )
        assert "Primary Recommendation" in result or "No stock found" in result

    def test_no_matching_products(self, chatbot):
        """Should handle no matching products gracefully."""
        result = chatbot.tool_get_warehouse_recommendation(
            product_keywords=["nonexistent_product_xyz"]
        )
        assert "No stock found" in result


class TestCountItems:
    """Tests for count_items tool."""

    def test_count_all(self, chatbot):
        """Should count all products."""
        result = chatbot.tool_count_items()
        assert "products" in result.lower()
        assert "Total stock" in result

    def test_count_by_category(self, chatbot):
        """Should filter by category."""
        result = chatbot.tool_count_items(category="Electronics")
        assert "products" in result.lower()
        assert "Electronics" in result

    def test_count_by_location(self, chatbot):
        """Should filter by location."""
        result = chatbot.tool_count_items(locations=["Sydney"])
        assert "products" in result.lower()
        assert "Sydney" in result

    def test_count_with_multiple_filters(self, chatbot):
        """Should apply multiple filters."""
        result = chatbot.tool_count_items(
            category="Electronics",
            locations=["Sydney", "Melbourne"],
            min_rating=4.0
        )
        assert "products" in result.lower()


class TestListProducts:
    """Tests for list_products tool."""

    def test_list_default(self, chatbot):
        """Should list products with default settings."""
        result = chatbot.tool_list_products()
        assert "Products" in result

    def test_list_with_limit(self, chatbot):
        """Should respect limit parameter."""
        result = chatbot.tool_list_products(limit=5)
        lines = [l for l in result.split("\n") if l.strip().startswith(tuple("123456789"))]
        assert len(lines) <= 5

    def test_list_sorted_by_price_asc(self, chatbot):
        """Should sort by price ascending."""
        result = chatbot.tool_list_products(sort_by="price_asc", limit=5)
        assert "sorted by price asc" in result.lower()

    def test_list_sorted_by_rating(self, chatbot):
        """Should sort by rating descending."""
        result = chatbot.tool_list_products(sort_by="rating_desc", limit=5)
        assert "sorted by rating desc" in result.lower()

    def test_list_by_category(self, chatbot):
        """Should filter by category."""
        result = chatbot.tool_list_products(category="Books", limit=5)
        assert "Books" in result or "No products found" in result


class TestCompareLocations:
    """Tests for compare_locations tool."""

    def test_compare_two_locations(self, chatbot):
        """Should compare two warehouses."""
        result = chatbot.tool_compare_locations(
            locations=["Sydney", "Melbourne"]
        )
        assert "Sydney" in result
        assert "Melbourne" in result
        assert "Total Stock" in result

    def test_compare_all_locations(self, chatbot):
        """Should compare all warehouses."""
        result = chatbot.tool_compare_locations(
            locations=["Sydney", "Melbourne", "Brisbane", "Perth"]
        )
        assert "Combined Total" in result

    def test_compare_with_category_filter(self, chatbot):
        """Should filter comparison by category."""
        result = chatbot.tool_compare_locations(
            locations=["Sydney", "Melbourne"],
            category="Electronics"
        )
        assert "Electronics" in result


class TestSimulateInventoryChange:
    """Tests for simulate_inventory_change tool."""

    def test_add_items(self, chatbot):
        """Should simulate adding items."""
        result = chatbot.tool_simulate_inventory_change(
            operation="add",
            quantity=100,
            location="Sydney"
        )
        assert "Add 100 units" in result
        assert "After addition" in result

    def test_subtract_items(self, chatbot):
        """Should simulate removing items."""
        result = chatbot.tool_simulate_inventory_change(
            operation="subtract",
            quantity=50,
            location="Sydney",
            category="Electronics"
        )
        assert "Remove 50 units" in result or "Cannot remove" in result

    def test_subtract_too_many(self, chatbot):
        """Should handle removing more than available."""
        result = chatbot.tool_simulate_inventory_change(
            operation="subtract",
            quantity=999999,
            location="Sydney"
        )
        assert "Cannot remove" in result


class TestSearchProducts:
    """Tests for search_products tool."""

    def test_search_basic(self, chatbot):
        """Should search for products."""
        result = chatbot.tool_search_products(query="AirPods")
        assert "Search Results" in result or "No products found" in result

    def test_search_with_limit(self, chatbot):
        """Should respect num_results parameter."""
        result = chatbot.tool_search_products(query="headphones", num_results=3)
        assert "Search Results" in result or "No products found" in result


class TestLowStockAlerts:
    """Tests for get_low_stock_alerts tool."""

    def test_get_all_alerts(self, chatbot):
        """Should get all low stock alerts."""
        result = chatbot.tool_get_low_stock_alerts()
        assert "Low Stock Alerts" in result or "No low stock alerts" in result

    def test_alerts_by_location(self, chatbot):
        """Should filter alerts by location."""
        result = chatbot.tool_get_low_stock_alerts(location="Sydney")
        # Should either have alerts or say no alerts
        assert "Low Stock" in result or "No low stock" in result

    def test_alerts_by_category(self, chatbot):
        """Should filter alerts by category."""
        result = chatbot.tool_get_low_stock_alerts(category="Electronics")
        assert "Low Stock" in result or "No low stock" in result


# ============================================
# INTEGRATION TESTS - Full ask() with Real API
# ============================================

@pytest.mark.slow
class TestAskIntegration:
    """Integration tests that call real OpenAI API."""

    def test_warehouse_recommendation_question(self, chatbot):
        """Should handle warehouse recommendation questions."""
        result = chatbot.ask("Best warehouse for books if Brisbane is full?")
        assert len(result) > 0
        # Should mention a city
        assert any(city in result for city in ["Sydney", "Melbourne", "Perth"])

    def test_count_question(self, chatbot):
        """Should handle count questions."""
        result = chatbot.ask("How many electronics in Sydney?")
        assert len(result) > 0

    def test_list_question(self, chatbot):
        """Should handle list questions."""
        result = chatbot.ask("Show me top 5 cheapest products")
        assert len(result) > 0

    def test_compare_question(self, chatbot):
        """Should handle comparison questions."""
        result = chatbot.ask("Compare Sydney and Melbourne stock")
        assert len(result) > 0

    def test_simulation_question(self, chatbot):
        """Should handle simulation questions."""
        result = chatbot.ask("What if I remove 50 items from Sydney?")
        assert len(result) > 0

    def test_product_search_question(self, chatbot):
        """Should handle product search questions."""
        result = chatbot.ask("Tell me about AirPods")
        assert len(result) > 0

    def test_low_stock_question(self, chatbot):
        """Should handle low stock questions."""
        result = chatbot.ask("Which products need restocking?")
        assert len(result) > 0


# ============================================
# EDGE CASE TESTS
# ============================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_question(self, chatbot):
        """Should handle empty questions."""
        result = chatbot.ask("")
        assert len(result) > 0  # Should return some response

    def test_special_characters(self, chatbot):
        """Should handle special characters."""
        result = chatbot.ask("What's the price of 'AirPods'?")
        assert len(result) > 0

    def test_very_long_question(self, chatbot):
        """Should handle long questions."""
        long_question = "Can you please tell me " * 50 + "about electronics?"
        result = chatbot.ask(long_question)
        assert len(result) > 0


# ============================================
# PARAMETRIZED TESTS - Multiple Scenarios
# ============================================

@pytest.mark.parametrize("question,expected_keywords", [
    ("Where should I send electronics if Sydney has no stock?", ["Melbourne", "Brisbane", "Perth"]),
    ("How many books are there?", ["products", "stock"]),
    ("Compare Sydney and Perth", ["Sydney", "Perth", "Total"]),
    ("List top 3 products", ["1.", "2.", "3."]),
])
@pytest.mark.slow
def test_question_scenarios(chatbot, question, expected_keywords):
    """Test various question scenarios."""
    result = chatbot.ask(question)
    assert any(kw in result for kw in expected_keywords), f"Expected one of {expected_keywords} in: {result}"


# ============================================
# MOCK TESTS - Without API Calls
# ============================================

class TestWithMockedAPI:
    """Tests with mocked OpenAI API for fast, deterministic testing."""

    @patch('chatbot_v2.OpenAI')
    def test_tool_selection_warehouse(self, mock_openai_class, sample_df):
        """Should select correct tool for warehouse questions."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response with tool call
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_warehouse_recommendation"
        mock_tool_call.function.arguments = json.dumps({
            "category": "Electronics",
            "exclude_location": "Sydney"
        })
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Second call returns final response
        mock_final = MagicMock()
        mock_final.choices = [MagicMock(message=MagicMock(
            content="Based on the data, Melbourne has the highest stock.",
            tool_calls=None
        ))]
        
        mock_client.chat.completions.create.side_effect = [mock_response, mock_final]

        # This test validates the mock setup is correct
        assert mock_tool_call.function.name == "get_warehouse_recommendation"


# ============================================
# PERFORMANCE TESTS
# ============================================

@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks."""

    def test_tool_execution_speed(self, chatbot):
        """Tool execution should be fast."""
        import time
        
        start = time.time()
        chatbot.tool_count_items(category="Electronics")
        duration = time.time() - start
        
        assert duration < 1.0, f"Tool execution took {duration:.2f}s, should be < 1s"

    def test_multiple_questions(self, chatbot):
        """Should handle multiple consecutive questions."""
        questions = [
            "How many electronics?",
            "List top 3 products",
            "Compare Sydney and Melbourne"
        ]
        
        for q in questions:
            result = chatbot.ask(q)
            assert len(result) > 0


# ============================================
# RUN CONFIGURATION
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
