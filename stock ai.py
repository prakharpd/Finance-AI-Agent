"""
Stock AI Agent - Gradio Dashboard
----------------------------------
Frontend (Gradio) wired to a real backend:
  - yfinance for live price / % change
  - OpenAI Agents SDK (pointed at a local Ollama model) for AI analysis

Setup:
    pip install gradio yfinance openai-agents

Run a local model with Ollama first, e.g.:
    ollama pull gemma3:27b
    ollama serve

Then:
    python stock_ai_dashboard.py
"""

import os
import re
import json
import asyncio

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("OPENAI_API_KEY", "ollama")

import yfinance as yf
import gradio as gr
from agents import Agent, Runner, function_tool, set_tracing_disabled

set_tracing_disabled(True)

MODEL_NAME = "gemma3:27b"   # <-- change to whatever tag you've pulled in Ollama

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "AMD", "JPM"]


# ---------------------------------------------------------------------------
# Backend: data + tools
# ---------------------------------------------------------------------------

def get_price_and_change(ticker: str):
    """Fetch current price and % change vs previous close. Never raises."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return None, None
        last_close = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else last_close
        change_pct = ((last_close - prev_close) / prev_close) * 100
        return round(float(last_close), 2), round(float(change_pct), 2)
    except Exception:
        return None, None


@function_tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    price, change = get_price_and_change(ticker)
    if price is None:
        return f"No price data found for {ticker}"
    return f"{ticker} is trading at ${price:.2f}, {change:+.2f}% vs previous close"


@function_tool
def get_analyst_recommendations(ticker: str) -> str:
    """Get recent analyst recommendations for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    if recs is None or recs.empty:
        return f"No analyst recommendations found for {ticker}"
    return recs.tail(10).to_string()


ANALYSIS_INSTRUCTIONS = """You are a financial analysis assistant.
When asked to analyze a ticker, use your tools to check the price and analyst
recommendations, then respond with STRICT JSON ONLY (no markdown fences, no
preamble) matching exactly this schema:

{
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "summary": "2-3 sentence overview of the stock's current situation",
  "key_points": ["point 1", "point 2", "point 3", "point 4"],
  "price_target": <number>,
  "price_target_pct": <number, percent upside/downside vs current price>
}

For free-form questions (not a full ticker analysis), respond with STRICT JSON:
{ "answer": "your answer here" }
"""

finance_agent = Agent(
    name="Finance Agent",
    instructions=ANALYSIS_INSTRUCTIONS,
    model=MODEL_NAME,
    tools=[get_stock_price, get_analyst_recommendations],
)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


async def run_ticker_analysis(ticker: str) -> dict:
    price, change = get_price_and_change(ticker)
    prompt = f"Analyze {ticker} as an investment right now."
    result = await Runner.run(finance_agent, prompt)
    try:
        data = _extract_json(result.final_output)
    except Exception:
        data = {
            "sentiment": "Neutral",
            "summary": result.final_output,
            "key_points": [],
            "price_target": None,
            "price_target_pct": None,
        }
    data["price"] = price
    data["change"] = change
    return data


async def run_free_question(question: str, ticker: str) -> str:
    prompt = f"Context ticker: {ticker}. Question: {question}"
    result = await Runner.run(finance_agent, prompt)
    try:
        data = _extract_json(result.final_output)
        return data.get("answer", result.final_output)
    except Exception:
        return result.final_output


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def sentiment_badge(sentiment: str) -> str:
    colors = {
        "Bullish": ("#e8f7ee", "#1e8a4c"),
        "Bearish": ("#fdecec", "#c0392b"),
        "Neutral": ("#f0f0f0", "#555555"),
    }
    bg, fg = colors.get(sentiment, colors["Neutral"])
    return f"""<span style="background:{bg};color:{fg};padding:4px 12px;
    border-radius:14px;font-size:13px;font-weight:600;">{sentiment}</span>"""


def render_analysis(ticker: str, data: dict) -> str:
    price = data.get("price")
    change = data.get("change")
    change_color = "#1e8a4c" if (change or 0) >= 0 else "#c0392b"
    change_str = f"{change:+.2f}%" if change is not None else "—"
    price_str = f"${price:.2f}" if price is not None else "—"

    key_points = "".join(
        f'<li style="margin-bottom:6px;">{point}</li>' for point in data.get("key_points", [])
    ) or "<li>No key points available</li>"

    target = data.get("price_target")
    target_pct = data.get("price_target_pct")
    target_str = f"${target:,.2f}" if isinstance(target, (int, float)) else "N/A"
    target_pct_str = f"({target_pct:+.1f}%)" if isinstance(target_pct, (int, float)) else ""

    return f"""
<div style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;">
  <div style="display:flex;align-items:center;gap:12px;padding:16px 20px;
              background:#fafafa;border-bottom:1px solid #eee;">
    <div style="width:34px;height:34px;border-radius:50%;background:#f97316;color:white;
                display:flex;align-items:center;justify-content:center;font-weight:700;">
      {ticker[0]}
    </div>
    <div style="font-weight:700;font-size:16px;">{ticker}</div>
    <div style="color:#777;">{price_str}</div>
    <div style="color:{change_color};font-weight:600;">{change_str}</div>
    <div style="margin-left:auto;">{sentiment_badge(data.get('sentiment', 'Neutral'))}</div>
  </div>
  <div style="padding:18px 20px;font-size:15px;line-height:1.6;">
    {data.get('summary', '')}
  </div>
  <div style="padding:0 20px 18px 20px;">
    <div style="color:#777;font-size:13px;margin-bottom:6px;">Key Points</div>
    <ul style="padding-left:18px;font-size:14px;">{key_points}</ul>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:14px 20px;border-top:1px solid #eee;background:#fafafa;">
    <div style="color:#777;font-size:13px;">12-month price target</div>
    <div><b>{target_str}</b> <span style="color:#1e8a4c;">{target_pct_str}</span></div>
  </div>
</div>
"""


def ticker_button_label(ticker: str) -> str:
    price, change = get_price_and_change(ticker)
    if price is None:
        return f"{ticker}\n—"
    arrow = "▲" if change >= 0 else "▼"
    return f"{ticker}\n{arrow} {change:+.2f}%"


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

async def on_select_ticker(ticker: str):
    data = await run_ticker_analysis(ticker)
    html = render_analysis(ticker, data)
    return html, ticker, gr.update(value=f"Analyze {ticker}")


async def on_analyze_click(ticker: str):
    return await on_select_ticker(ticker)


async def on_ask(question: str, ticker: str, history):
    if not question.strip():
        return history, ""
    answer = await run_free_question(question, ticker)
    history = history + [[question, answer]]
    return history, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container {background:#f7f8fa !important;}
#header {display:flex;align-items:center;justify-content:space-between;
         padding:10px 4px;border-bottom:1px solid #e5e7eb;margin-bottom:14px;}
.sidebar-card {background:white;border:1px solid #e5e7eb;border-radius:10px;
               padding:16px;margin-bottom:16px;}
.ticker-btn {text-align:left !important;}
"""

with gr.Blocks(css=CSS, title="Stock AI Agent") as demo:
    selected_ticker = gr.State("AAPL")

    with gr.Row(elem_id="header"):
        gr.Markdown("### 🟧 **Stock AI Agent**  `Demo`")
        gr.Markdown(f"Powered by `{MODEL_NAME}`")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("**Select a Ticker**  \n*Click to run AI analysis*")
                ticker_buttons = {}
                rows = [TICKERS[i:i + 3] for i in range(0, len(TICKERS), 3)]
                for row in rows:
                    with gr.Row():
                        for t in row:
                            btn = gr.Button(ticker_button_label(t), elem_classes="ticker-btn")
                            ticker_buttons[t] = btn

            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("**Agent Config**")
                gr.Markdown(
                    f"""
                    Model&nbsp;&nbsp;&nbsp;&nbsp;`{MODEL_NAME}`  
                    Tools&nbsp;&nbsp;&nbsp;&nbsp;`get_stock_price, get_analyst_recommendations`  
                    Mode&nbsp;&nbsp;&nbsp;&nbsp;`analysis`  
                    Memory&nbsp;&nbsp;`stateless`
                    """
                )

        with gr.Column(scale=2):
            with gr.Row():
                analyze_btn = gr.Button("Analyze AAPL", variant="primary")
            analysis_output = gr.HTML(value="Select a ticker to begin.")

            chatbot = gr.Chatbot(label=None, height=200)
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask about a stock, e.g. 'What's the outlook for NVDA?'",
                    show_label=False,
                    scale=8,
                )
                send_btn = gr.Button("➤", scale=1)

    gr.Markdown(
        "<div style='text-align:center;color:#999;font-size:12px;'>"
        "AI-generated analysis — not financial advice.</div>"
    )

    # Wire up ticker buttons
    for t, btn in ticker_buttons.items():
        btn.click(
            fn=on_select_ticker,
            inputs=gr.State(t),
            outputs=[analysis_output, selected_ticker, analyze_btn],
        )

    analyze_btn.click(
        fn=on_analyze_click,
        inputs=selected_ticker,
        outputs=[analysis_output, selected_ticker, analyze_btn],
    )

    send_btn.click(
        fn=on_ask,
        inputs=[question_box, selected_ticker, chatbot],
        outputs=[chatbot, question_box],
    )
    question_box.submit(
        fn=on_ask,
        inputs=[question_box, selected_ticker, chatbot],
        outputs=[chatbot, question_box],
    )

    demo.load(fn=on_select_ticker, inputs=gr.State("AAPL"),
              outputs=[analysis_output, selected_ticker, analyze_btn])


if __name__ == "__main__":
    demo.launch()