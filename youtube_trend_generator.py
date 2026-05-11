"""
YouTube Trend Video Generator
Step-by-step tool to discover trending content and generate video ideas
based on your own YouTube channel using the YouTube Data API v3.
"""

import os
import streamlit as st
import pandas as pd
import re
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Load .env automatically (before anything else) ──────────────────────────
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

# ── Optional deps (installed via requirements.txt) ──────────────────────────
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YT_SDK_AVAILABLE = True
except ImportError:
    YT_SDK_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# ── Env-sourced defaults (pre-fill sidebar without user typing) ──────────────
_ENV_YT_KEY      = os.environ.get("YOUTUBE_API_KEY", "")
_ENV_CHANNEL_ID  = os.environ.get("YOUTUBE_CHANNEL_ID", "")
_ENV_CLAUDE_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
_ENV_REGION      = os.environ.get("DEFAULT_REGION", "United States")
_ENV_CATEGORY    = os.environ.get("DEFAULT_CATEGORY", "All")
_ENV_MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "25"))

# ── Constants ────────────────────────────────────────────────────────────────
CATEGORIES = {
    "All": "0",
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Gaming": "20",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
    "News & Politics": "25",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "Travel & Events": "19",
    "Nonprofits & Activism": "29",
}

REGIONS = {
    "United States": "US",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "India": "IN",
    "Germany": "DE",
    "France": "FR",
    "Japan": "JP",
    "Brazil": "BR",
    "Mexico": "MX",
    "South Korea": "KR",
    "Indonesia": "ID",
    "Philippines": "PH",
    "Singapore": "SG",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_duration(iso: str) -> int:
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not match:
        return 0
    h, m, s = (int(x or 0) for x in match.groups())
    return h * 3600 + m * 60 + s


def fmt_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def fmt_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def engagement_rate(views: int, likes: int, comments: int) -> float:
    if views == 0:
        return 0.0
    return round((likes + comments) / views * 100, 2)


def days_since(published_at: str) -> int:
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return max(1, (datetime.now(timezone.utc) - pub).days)
    except Exception:
        return 1


def extract_title_patterns(titles: list[str]) -> dict:
    patterns = {
        "numbers": 0,
        "questions": 0,
        "caps_words": 0,
        "parentheses": 0,
        "colons": 0,
        "ellipsis": 0,
    }
    for t in titles:
        if re.search(r"\b\d+\b", t):
            patterns["numbers"] += 1
        if "?" in t:
            patterns["questions"] += 1
        if re.search(r"\b[A-Z]{2,}\b", t):
            patterns["caps_words"] += 1
        if re.search(r"[\(\[]", t):
            patterns["parentheses"] += 1
        if ":" in t:
            patterns["colons"] += 1
        if "..." in t or "…" in t:
            patterns["ellipsis"] += 1
    total = len(titles)
    return {k: round(v / total * 100) for k, v in patterns.items()}


def top_keywords(titles: list[str], n: int = 20) -> list[tuple[str, int]]:
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "it", "this", "that", "i", "my", "you",
        "we", "he", "she", "they", "are", "was", "be", "do", "did", "get",
        "got", "how", "what", "why", "when", "your", "its", "not", "no",
        "s", "t", "re", "ve", "ll", "d", "has", "have", "had",
    }
    words = []
    for t in titles:
        words.extend(
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]+\b", t)
            if w.lower() not in stop and len(w) > 2
        )
    return Counter(words).most_common(n)


# ── YouTube API calls ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trending(api_key: str, region: str, category_id: str, max_results: int = 50) -> list[dict]:
    yt = build("youtube", "v3", developerKey=api_key)
    params = dict(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=region,
        maxResults=max_results,
        videoCategoryId=category_id,
    )
    if category_id == "0":
        del params["videoCategoryId"]
    resp = yt.videos().list(**params).execute()
    items = resp.get("items", [])
    results = []
    for v in items:
        sn = v["snippet"]
        st_ = v.get("statistics", {})
        cd = v.get("contentDetails", {})
        results.append({
            "video_id":    v["id"],
            "title":       sn.get("title", ""),
            "channel":     sn.get("channelTitle", ""),
            "channel_id":  sn.get("channelId", ""),
            "published_at": sn.get("publishedAt", ""),
            "description": sn.get("description", "")[:300],
            "tags":        sn.get("tags", []),
            "category_id": sn.get("categoryId", ""),
            "thumbnail":   sn.get("thumbnails", {}).get("high", {}).get("url", ""),
            "views":       int(st_.get("viewCount", 0)),
            "likes":       int(st_.get("likeCount", 0)),
            "comments":    int(st_.get("commentCount", 0)),
            "duration_s":  parse_duration(cd.get("duration", "")),
            "url":         f"https://www.youtube.com/watch?v={v['id']}",
        })
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_channel_info(api_key: str, channel_id: str) -> dict:
    yt = build("youtube", "v3", developerKey=api_key)
    resp = yt.channels().list(
        part="snippet,statistics,brandingSettings",
        id=channel_id,
    ).execute()
    items = resp.get("items", [])
    if not items:
        return {}
    ch = items[0]
    sn = ch["snippet"]
    st_ = ch.get("statistics", {})
    return {
        "name":        sn.get("title", ""),
        "description": sn.get("description", "")[:400],
        "country":     sn.get("country", "N/A"),
        "subscribers": int(st_.get("subscriberCount", 0)),
        "total_views": int(st_.get("viewCount", 0)),
        "video_count": int(st_.get("videoCount", 0)),
        "thumbnail":   sn.get("thumbnails", {}).get("default", {}).get("url", ""),
    }


# ── Claude-powered script generator ─────────────────────────────────────────

def generate_script_with_claude(
    api_key: str,
    channel_info: dict,
    video_title: str,
    trend_data: dict,
) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are a professional YouTube content strategist and scriptwriter.

Channel context:
- Channel name: {channel_info.get('name', 'My Channel')}
- Niche/description: {channel_info.get('description', 'General content')}
- Subscribers: {fmt_number(channel_info.get('subscribers', 0))}

Trending video inspiration:
- Title: {video_title}
- Trend keywords: {', '.join(trend_data.get('top_keywords', [])[:10])}
- Avg duration of trending videos: {trend_data.get('avg_duration', '8-12 minutes')}

Task: Generate a COMPLETE, ready-to-film video script outline for my channel that is INSPIRED BY (not copying) the trending video above. The script should:
1. Fit my channel's style and audience
2. Use the trending topic angle but with original content
3. Include: Hook (first 30s), Intro, 3-5 main sections, CTA, Outro
4. Suggest specific B-roll shots and on-screen text for each section
5. Include SEO-optimized title (5 variations), description (500 chars), and 15 tags

Format clearly with headers for each section."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_titles_from_patterns(
    trend_keywords: list[str],
    patterns: dict,
    channel_niche: str,
    original_title: str,
) -> list[str]:
    kw = trend_keywords[:5] if trend_keywords else ["this topic"]
    topic = kw[0].title()
    suggestions = [
        f"I Tried {topic} For 30 Days (Here's What Happened)",
        f"{topic}: Everything You Need To Know In 2025",
        f"Why Everyone Is Talking About {topic} Right Now",
        f"The TRUTH About {topic} Nobody Tells You",
        f"{topic} Explained: Beginner To Pro In {kw[1].title() if len(kw) > 1 else '10'} Minutes",
        f"How I Used {topic} To Change Everything",
        f"Is {topic} Worth It? My Honest Review",
        f"{topic} vs {kw[2].title() if len(kw) > 2 else 'Everything Else'}: Which Is Better?",
    ]
    return suggestions


# ── Main Streamlit App ───────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="YouTube Trend Generator",
        page_icon="▶",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("▶ YouTube Trend Generator")
        st.markdown("---")

        st.subheader("Step 1 — API Credentials")
        _yt_hint = "✅ Loaded from .env" if _ENV_YT_KEY else "AIza..."
        yt_api_key = st.text_input(
            "YouTube Data API v3 Key",
            value=_ENV_YT_KEY,
            type="password",
            placeholder=_yt_hint,
            help="Get yours free at console.cloud.google.com → Enable YouTube Data API v3",
        )

        _ch_hint = "✅ Loaded from .env" if _ENV_CHANNEL_ID else "UCxxxxxxxxxxxxxxxxxx"
        channel_id = st.text_input(
            "Your YouTube Channel ID",
            value=_ENV_CHANNEL_ID,
            placeholder=_ch_hint,
            help="Found at: YouTube Studio → Customization → Basic info → Channel URL",
        )

        st.subheader("Step 2 — Trend Settings")
        _region_default = next(
            (name for name, code in REGIONS.items() if code == _ENV_REGION),
            list(REGIONS.keys())[0],
        )
        region_name = st.selectbox(
            "Region", list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(_region_default),
        )
        region_code = REGIONS[region_name]

        _cat_default = _ENV_CATEGORY if _ENV_CATEGORY in CATEGORIES else "All"
        category_name = st.selectbox(
            "Category", list(CATEGORIES.keys()),
            index=list(CATEGORIES.keys()).index(_cat_default),
        )
        category_id = CATEGORIES[category_name]

        max_results = st.slider("Videos to analyse", 10, 50, _ENV_MAX_RESULTS, 5)

        st.markdown("---")
        st.subheader("Step 3 — AI Script (Optional)")
        _claude_hint = "✅ Loaded from .env" if _ENV_CLAUDE_KEY else "sk-ant-..."
        claude_api_key = st.text_input(
            "Anthropic Claude API Key",
            value=_ENV_CLAUDE_KEY,
            type="password",
            placeholder=_claude_hint,
            help="Optional. Enables AI-powered script generation. Get at console.anthropic.com",
        )

        if _ENV_YT_KEY and _ENV_CHANNEL_ID:
            st.success("Credentials loaded from .env — ready to fetch!")

        st.markdown("---")
        fetch_btn = st.button("Fetch Trending Videos", type="primary", use_container_width=True)

    if not YT_SDK_AVAILABLE:
        st.error(
            "Missing dependency: `google-api-python-client`.\n\n"
            "Run: `pip install google-api-python-client google-auth`"
        )
        st.stop()

    if not fetch_btn and not st.session_state.get("videos"):
        _render_setup_guide()
        st.stop()

    if fetch_btn:
        if not yt_api_key:
            st.sidebar.error("YouTube API Key required.")
            st.stop()
        with st.spinner("Fetching trending videos…"):
            try:
                videos = fetch_trending(yt_api_key, region_code, category_id, max_results)
                st.session_state["videos"] = videos
                st.session_state["yt_api_key"] = yt_api_key
                st.session_state["channel_id"] = channel_id
                st.session_state["claude_api_key"] = claude_api_key
                if channel_id:
                    ch_info = fetch_channel_info(yt_api_key, channel_id)
                    st.session_state["channel_info"] = ch_info
                else:
                    st.session_state["channel_info"] = {}
            except HttpError as e:
                st.error(f"YouTube API error: {e.reason}")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    videos: list[dict] = st.session_state.get("videos", [])
    channel_info: dict = st.session_state.get("channel_info", {})

    if not videos:
        st.warning("No videos found. Try a different region or category.")
        st.stop()

    df = pd.DataFrame(videos)
    df["engagement"] = df.apply(
        lambda r: engagement_rate(r["views"], r["likes"], r["comments"]), axis=1
    )
    df["days_old"] = df["published_at"].apply(days_since)
    df["views_per_day"] = (df["views"] / df["days_old"]).astype(int)
    df["duration_fmt"] = df["duration_s"].apply(fmt_duration)

    titles = df["title"].tolist()
    patterns = extract_title_patterns(titles)
    keywords = top_keywords(titles)
    top_kw_list = [w for w, _ in keywords]

    all_tags = [tag for tags in df["tags"] for tag in tags]
    top_tags = [t for t, _ in Counter(all_tags).most_common(25)]

    avg_dur = int(df["duration_s"].mean())
    avg_views = int(df["views"].mean())
    avg_eng = round(df["engagement"].mean(), 2)

    trend_data = {
        "top_keywords": top_kw_list,
        "avg_duration": fmt_duration(avg_dur),
        "avg_views": avg_views,
        "avg_engagement": avg_eng,
        "patterns": patterns,
        "top_tags": top_tags,
    }

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "\U0001f4ca Trend Dashboard",
        "\U0001f525 Top Videos",
        "\U0001f4dd Content Ideas",
        "\U0001f3ac Script Generator",
        "\U0001f4cb My Channel",
    ])

    with tab1:
        st.header("Trend Dashboard")
        st.caption(f"Analysed {len(videos)} trending videos · {region_name} · {category_name}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Views", fmt_number(avg_views))
        c2.metric("Avg Engagement", f"{avg_eng}%")
        c3.metric("Avg Duration", fmt_duration(avg_dur))
        c4.metric("Videos Analysed", len(videos))
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Title Patterns (% of trending videos)")
            pat_df = pd.DataFrame(
                [{"Pattern": k.replace("_", " ").title(), "Usage %": v}
                 for k, v in patterns.items()]
            ).sort_values("Usage %", ascending=False)
            st.bar_chart(pat_df.set_index("Pattern"))
        with col_r:
            st.subheader("Top Keywords in Trending Titles")
            kw_df = pd.DataFrame(keywords[:15], columns=["Keyword", "Count"])
            st.bar_chart(kw_df.set_index("Keyword"))
        st.subheader("Optimal Video Duration Range")
        dur_bins = pd.cut(
            df["duration_s"] / 60,
            bins=[0, 3, 7, 12, 20, 30, 60, 999],
            labels=["<3m", "3-7m", "7-12m", "12-20m", "20-30m", "30-60m", "60m+"],
        )
        dur_count = dur_bins.value_counts().sort_index()
        st.bar_chart(dur_count)
        sweet_spot = dur_count.idxmax()
        st.info(f"Sweet-spot duration for {category_name} in {region_name}: **{sweet_spot}**")
        st.subheader("Top Tags Across Trending Videos")
        st.write(" ".join(f"`{t}`" for t in top_tags))

    with tab2:
        st.header("Top Trending Videos")
        sort_by = st.selectbox(
            "Sort by",
            ["views", "engagement", "views_per_day", "likes", "comments"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        sorted_df = df.sort_values(sort_by, ascending=False).head(20)
        for _, row in sorted_df.iterrows():
            with st.expander(f"▶ {row['title'][:80]}"):
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    if row["thumbnail"]:
                        st.image(row["thumbnail"], use_container_width=True)
                with col_info:
                    st.markdown(f"**Channel:** {row['channel']}")
                    st.markdown(
                        f"**Views:** {fmt_number(row['views'])} · "
                        f"**Likes:** {fmt_number(row['likes'])} · "
                        f"**Comments:** {fmt_number(row['comments'])}"
                    )
                    st.markdown(
                        f"**Engagement:** {row['engagement']}% · "
                        f"**Duration:** {row['duration_fmt']} · "
                        f"**Views/day:** {fmt_number(row['views_per_day'])}"
                    )
                    if row["tags"]:
                        st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in row["tags"][:10]))
                    st.markdown(f"[Open on YouTube]({row['url']})")
                    st.session_state.setdefault("selected_video", None)
                    if st.button("Use as inspiration →", key=f"sel_{row['video_id']}"):
                        st.session_state["selected_video"] = row.to_dict()
                        st.success("Video selected! Go to **Script Generator** tab.")

    with tab3:
        st.header("Content Ideas Generator")
        st.write(
            "These ideas are **inspired by** trending patterns — not copies. "
            "Adapt them to your channel's unique voice."
        )
        niche = st.text_input(
            "Describe your channel niche (helps tailor suggestions)",
            value=channel_info.get("description", "")[:100] if channel_info else "",
            placeholder="e.g. personal finance tips for millennials",
        )
        if st.button("Generate Ideas", type="primary"):
            ideas = generate_titles_from_patterns(top_kw_list, patterns, niche, "")
            st.subheader("Suggested Video Titles")
            for i, idea in enumerate(ideas, 1):
                st.markdown(f"**{i}.** {idea}")
            st.markdown("---")
            st.subheader("SEO Tags to Use")
            combined_tags = list(dict.fromkeys(top_tags + top_kw_list))[:25]
            st.write(", ".join(combined_tags))
            st.markdown("---")
            st.subheader("Content Angle Recommendations")
            angle_tips = {
                "Numbers": (patterns["numbers"] > 50, "Use numbered lists in your title (e.g. '7 Ways to…')"),
                "Questions": (patterns["questions"] > 30, "Frame your title as a question to spark curiosity"),
                "ALL CAPS words": (patterns["caps_words"] > 40, "Capitalise 1-2 power words to add emphasis"),
                "Colons": (patterns["colons"] > 40, "Use a colon to add subtitle context (e.g. 'Topic: Why It Matters')"),
                "Parentheses": (patterns["parentheses"] > 30, "Add context in brackets (e.g. '(My Honest Review)')"),
            }
            for label, (active, tip) in angle_tips.items():
                icon = "✅" if active else "⬜"
                st.markdown(f"{icon} **{label}** — {tip}")

    with tab4:
        st.header("Script Generator")
        sel = st.session_state.get("selected_video")
        if not sel:
            st.info("Go to **Top Videos** tab and click **Use as inspiration →** on a video, then come back here.")
        else:
            st.markdown(f"**Inspiration video:** [{sel['title']}]({sel['url']})")
            st.markdown(
                f"Views: {fmt_number(sel['views'])} · "
                f"Engagement: {sel['engagement']}% · "
                f"Duration: {sel['duration_fmt']}"
            )
            st.markdown("---")
            use_claude = (
                CLAUDE_AVAILABLE
                and st.session_state.get("claude_api_key")
                and channel_info
            )
            if use_claude:
                st.success("Claude AI is available — full script will be generated.")
                if st.button("Generate Full Script with Claude AI", type="primary"):
                    with st.spinner("Claude is writing your script…"):
                        try:
                            script = generate_script_with_claude(
                                st.session_state["claude_api_key"],
                                channel_info,
                                sel["title"],
                                trend_data,
                            )
                            st.session_state["last_script"] = script
                        except Exception as e:
                            st.error(f"Claude error: {e}")
            else:
                if not CLAUDE_AVAILABLE or not st.session_state.get("claude_api_key"):
                    st.info("Add a Claude API key in the sidebar to enable AI script generation.")
                if not channel_info:
                    st.info("Enter your Channel ID in the sidebar to personalise the script.")
                if st.button("Generate Template Script", type="primary"):
                    st.session_state["last_script"] = _build_template_script(sel, trend_data, channel_info)
            if "last_script" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state["last_script"])
                st.download_button(
                    "Download Script (.txt)",
                    data=st.session_state["last_script"],
                    file_name=f"script_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                )

    with tab5:
        st.header("My Channel Overview")
        if not channel_info:
            st.warning("Enter your Channel ID in the sidebar and click **Fetch Trending Videos**.")
        else:
            col_a, col_b = st.columns([1, 4])
            with col_a:
                if channel_info.get("thumbnail"):
                    st.image(channel_info["thumbnail"], width=100)
            with col_b:
                st.subheader(channel_info["name"])
                st.caption(channel_info.get("description", ""))
            m1, m2, m3 = st.columns(3)
            m1.metric("Subscribers", fmt_number(channel_info["subscribers"]))
            m2.metric("Total Views", fmt_number(channel_info["total_views"]))
            m3.metric("Videos", channel_info["video_count"])
            st.markdown("---")
            st.subheader("Gap Analysis: You vs. Trending")
            st.markdown(
                f"- Trending avg views/video: **{fmt_number(avg_views)}**\n"
                f"- Trending avg engagement: **{avg_eng}%**\n"
                f"- Your subscribers: **{fmt_number(channel_info['subscribers'])}**\n"
                f"- Sweet-spot video length: **{fmt_duration(avg_dur)}**\n\n"
                "**Action plan:**\n"
                f"1. Target the `{top_kw_list[0] if top_kw_list else 'trending'}` keyword in your next video.\n"
                f"2. Aim for a **{fmt_duration(avg_dur)}** video — that's the trending sweet-spot.\n"
                f"3. Use title patterns: "
                + ", ".join(k for k, v in sorted(patterns.items(), key=lambda x: -x[1])[:3])
                + ".\n"
                f"4. Include these tags: {', '.join(top_tags[:8])}."
            )


def _build_template_script(sel: dict, trend_data: dict, channel_info: dict) -> str:
    kw = trend_data["top_keywords"][:5]
    ch_name = channel_info.get("name", "My Channel") if channel_info else "My Channel"
    topic = kw[0].title() if kw else "this topic"
    title_ideas = generate_titles_from_patterns(kw, trend_data["patterns"], "", "")
    tags_str = ", ".join(trend_data["top_tags"][:15])

    return f"""# VIDEO SCRIPT TEMPLATE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Channel: {ch_name}
Inspiration: {sel['title']}

===============================================
TITLE SUGGESTIONS (pick one, make it yours)
===============================================
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(title_ideas))}

===============================================
SEO DESCRIPTION (edit before publishing)
===============================================
In this video, I cover everything you need to know about {topic}.
Whether you're a beginner or have experience, this breakdown will help you
understand the key concepts and take action right away.

Chapters:
0:00 - Intro & Hook
1:30 - What Is {topic}?
3:00 - Why It Matters Right Now
6:00 - Step-by-Step Breakdown
10:00 - Common Mistakes to Avoid
13:00 - My Honest Take
14:30 - Outro & Next Steps

Subscribe for weekly videos on {', '.join(kw[:3])}.

Tags: {tags_str}

===============================================
SCRIPT
===============================================

-- HOOK (0:00-0:30) --
[On camera, energetic]
"In the next {trend_data['avg_duration']}, I'm going to show you exactly how
{topic} works - and why people who understand this are miles ahead.
Stay until the end, because I'll share the one mistake most people make."

[B-roll: quick montage of the topic in action]

-- INTRO (0:30-1:30) --
[On camera]
"Hey, welcome back - I'm [Your Name] from {ch_name}.
If you're new here, this channel is all about [your niche].
Hit subscribe so you don't miss the next one.

Today we're diving into {topic}.
Here's what we're covering: [briefly list 3 main points]."

-- SECTION 1: What Is {topic}? (1:30-3:00) --
[On camera + screen recording / slides]
"Let's start from the top. {topic.capitalize()} is basically..."
[Explain in plain language. Use an analogy.]
"Think of it like [relatable analogy]."

[B-roll: relevant footage or graphics]

-- SECTION 2: Why It Matters Right Now (3:00-6:00) --
[On camera]
"Here's why this is blowing up right now - [2-3 reasons with data or examples].
The numbers don't lie: [stat or example from trending video context]."

[Show on-screen text for key stats]

-- SECTION 3: Step-by-Step Breakdown (6:00-10:00) --
[Screen share or whiteboard]
"Alright, here's exactly how to [do the thing] in [X] steps."
Step 1: [Action] - [Why it works]
Step 2: [Action] - [Pro tip]
Step 3: [Action] - [Common pitfall to avoid]

-- SECTION 4: Common Mistakes (10:00-13:00) --
[On camera]
"Before you go do this, here are the mistakes I see all the time:"
Mistake 1: [Describe it] -> [How to fix it]
Mistake 2: [Describe it] -> [How to fix it]

-- MY HONEST TAKE (13:00-14:30) --
[Relaxed on camera]
"Here's my personal take after [experience]. {topic.capitalize()} is [your opinion].
If I had to give you one piece of advice: [memorable one-liner]."

-- OUTRO & CTA (14:30-end) --
[On camera, upbeat]
"That's it for today. If this helped you, smash that like button - it genuinely
helps this channel grow.

Comment below: what's YOUR experience with {topic}? I read every comment.

And if you want to go deeper, check out this video next -> [point to end screen]

See you in the next one. Peace."

[End screen: 2 video suggestions + Subscribe button, hold 20 seconds]

===============================================
THUMBNAIL CONCEPT
===============================================
- Background: bold single colour (red, yellow, or blue)
- Your face: surprised / excited expression, left side
- Text overlay: 2-3 words max, right side, large font
  Suggestion: "{topic.upper()}" or "FINALLY EXPLAINED"
- Optional: contrast icon or arrow pointing at your face
- NO more than 3 elements - keep it scannable at thumbnail size

===============================================
UPLOAD CHECKLIST
===============================================
[ ] Title uses keyword in first 3 words
[ ] Description has keyword in first 2 sentences
[ ] 10-15 tags added (mix broad + specific)
[ ] Custom thumbnail uploaded
[ ] End screens set (20-second hold)
[ ] Cards added at 20% and 60% of video
[ ] Chapters added to description
[ ] Premiere or scheduled publish (Thu-Sat 2-4pm local = peak hours)
[ ] Shared to community tab on publish
"""


def _render_setup_guide():
    st.title("▶ YouTube Trend Generator")
    st.subheader("Discover what's trending → create your own original video")
    st.markdown("""
This tool gives you a **step-by-step workflow** to find trending YouTube videos,
analyse the patterns that make them go viral, and generate an original script for
your own channel - without copying anyone.

---

## Quick-start Guide

### Step 1 - Get a Free YouTube Data API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use an existing one)
3. Navigate to **APIs & Services -> Library**
4. Search for **YouTube Data API v3** and click **Enable**
5. Go to **APIs & Services -> Credentials -> Create Credentials -> API Key**
6. Copy the key and paste it in the sidebar

> The free quota is **10,000 units/day** - each fetch uses ~50 units, so you
> get ~200 analyses per day at no cost.

---

### Step 2 - Find Your Channel ID
1. Open [YouTube Studio](https://studio.youtube.com/)
2. Click **Customization -> Basic info**
3. Scroll to **Channel URL** - your ID starts with `UC...`
4. Alternatively, go to your channel page and copy the last part of the URL

---

### Step 3 - (Optional) Add Claude API Key for AI Scripts
- Sign up at [console.anthropic.com](https://console.anthropic.com/)
- New accounts get free credits
- Without it, the tool uses a smart rule-based template instead

---

### Step 4 - Run the Generator
1. Fill in the sidebar fields
2. Choose your **Region** and **Category**
3. Click **Fetch Trending Videos**
4. Explore the tabs: Dashboard -> Top Videos -> Content Ideas -> Script Generator

---
    """)
    col1, col2, col3 = st.columns(3)
    col1.info("**Free to use**\nYouTube API has a generous free tier")
    col2.info("**No copying**\nIdeas are inspired by trends, not cloned")
    col3.info("**Your channel**\nPersonalised to your niche & audience")


if __name__ == "__main__":
    main()
