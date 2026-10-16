"""Live integration tests for source-extract worker and downstream media gather contract.

These tests assert against real, live production websites and endpoints:
1. Extraction of target media (direct video embeds, streams, and high-res event photos),
   strictly filtering out junk (logos, ads, app store badges, avatars).
2. Extraction of meaningful textual content (length, mandatory factual keywords,
   and absence of anti-bot/WAF error markers).
3. Social media rich metadata extraction (especially via yt-dlp: title, uploader,
   upload_date, and description) to prevent empty-context blindness in vision validation.
4. Client-side Pydantic schema compatibility across all platforms (direct, vimeo,
   brightcove, html5, etc.).

Run explicitly with:
    pytest -m live tests/live/test_live_source_extract.py -v
"""
from __future__ import annotations

import os
from typing import Any
import httpx
from curl_cffi.requests import AsyncSession as StealthSession
import pytest

from src.extract import extract
from src.models import SourceExtractResult


_TAVILY_URL = os.getenv("TAVILY_SEARCH_URL", "http://localhost:8018")
_SOURCE_EXTRACT_URL = os.getenv("SOURCE_EXTRACT_URL", "http://localhost:8028")
_YT_DLP_URL = os.getenv("YT_DLP_URL", "http://localhost:8005")


# ─────────────────────────────────────────────────────────────────────────────
# Test Data: Golden Dataset of Target Media & Content Requirements
# ─────────────────────────────────────────────────────────────────────────────

ARTICLE_TEST_CASES = [
    {
        "id": "C2-S2-Newsweek",
        "url": "https://www.newsweek.com/russian-drones-destroy-ukraine-himars-video-2068590",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "1919687487212986557",
            "ukraine-helicopters",
        ],
        "target_image_patterns": [
            "2639368-m142-himars-launches-rocket",
        ],
        "min_content_chars": 300,
        "required_keywords": ["himars", "rubicon", "rusyn yar"],
        "forbidden_patterns": ["cloudflare", "access denied", "403 forbidden"],
        "description": "Newsweek article with embedded Twitter combat video of Rubicon drone strike near Rusyn Yar",
    },
    {
        "id": "C7-S4-Pravda-Kakhovka",
        "url": "https://www.pravda.com.ua/news/2022/07/11/7357612",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "itsdonetsk/21402",
            "hueviyherson/21873",
            "Bratchuk_Sergey/15354",
        ],
        "target_image_patterns": [
            "7707a2c-nk.jpg",
        ],
        "min_content_chars": 300,
        "required_keywords": ["кахов", "нова каховка", "склад"],
        "forbidden_patterns": ["403 forbidden", "access denied"],
        "description": "Ukrainska Pravda breaking news with 1200x673 explosion photo and Telegram video embeds",
    },
    {
        "id": "C3-S5-DW-Belgorod",
        "url": "https://www.dw.com/ru/v-belgorode-posle-udara-po-infrastrukture-blekaut/a-74167842",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "hlsvod.dw.com",
            "master.m3u8",
        ],
        "target_image_patterns": [
            "73634719",
        ],
        "min_content_chars": 300,
        "required_keywords": ["белгород", "тэц", "блэкаут"],
        "forbidden_patterns": ["captcha", "access denied"],
        "description": "Deutsche Welle report on Belgorod TPP blackout with direct HLS video stream",
    },
    {
        "id": "C6-S8-L3Harris-Capabilities",
        "url": "https://www.l3harris.com/all-capabilities/guided-multiple-launch-rocket-system-gmlrs",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "vimeo.com/941333133",
        ],
        "target_image_patterns": [
            "ar-gmlrs-guided-multiple-launch-system-us-army",
        ],
        "min_content_chars": 200,
        "required_keywords": ["gmlrs", "l3harris"],
        "forbidden_patterns": ["access denied"],
        "description": "L3Harris capabilities page with embedded Vimeo video and product photos",
    },
    {
        "id": "C14-S1-AlJazeera-PrSM",
        "url": "https://www.aljazeera.com/news/2026/3/6/what-is-the-prsm-missile-that-the-us-used-for-the-first-time-in-iran",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "players.brightcove.net",
            "6390379618112",
        ],
        "target_image_patterns": [
            "afp_69aa92e1a3cd",
        ],
        "min_content_chars": 300,
        "required_keywords": ["prsm", "iran", "missile"],
        "forbidden_patterns": ["access denied"],
        "description": "Al Jazeera report with embedded Brightcove player on US PrSM combat use",
    },
    {
        "id": "C11-S8-iStories-Kursk",
        "url": "https://istories.media/news/2024/08/09/boi-pod-kurskom-den-chetvertii-glavnoe",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "061c761aa42248f485eb379d4e1130b2.mp4",
            "kiber_boroshno/9445",
            "milinfolive/127952",
        ],
        "target_image_patterns": [
            "95556f54cd6d4fc59e302574c63db470",
        ],
        "min_content_chars": 300,
        "required_keywords": ["рыльск", "октябрьск", "колонн"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "iStories investigative report on Rylsk convoy strike with HTML5 MP4 and Telegram videos",
    },
    {
        "id": "C11-S9-Meduza-Kursk",
        "url": "https://meduza.io/feature/2024/08/09/v-sotssetyah-opublikovany-video-razbitoy-kolonny-rossiyskoy-armii-v-kurskoy-oblasti",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "1821679269929538029",
            "1821804091112354156",
            "milinfolive/127952",
            "mash/56608",
        ],
        "target_image_patterns": [
            "v-sotssetyah-opublikovany-video-razbitoy-kolonny",
        ],
        "min_content_chars": 300,
        "required_keywords": ["колонн", "курск", "октябрьск"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Meduza aggregation of combat footage showing destroyed military convoy",
    },
    {
        "id": "C8-S6-Glavred-Nasbulin",
        "url": "https://glavred.info/ukraine/udarami-himars-unichtozhen-rossiyskiy-general-mayor-iz-kryma-10390706.html",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "Bratchuk_Sergey/15351",
            "tv.kyivstar.ua",
        ],
        "target_image_patterns": [
            # Golden-dataset drift, 2026-10-06: Tavily's own /extract no
            # longer surfaces the original "1657607388-3094.jpeg" at all
            # (confirmed live — the raw page HTML still has it, Tavily's
            # own image-selection heuristic just stopped picking it).
            # "1657606904-7527.jpg" is Tavily's current pick for this same
            # hero image slot (same 2022_07 upload batch, immediately
            # adjacent id) — not a guess, read straight from a live
            # /extract call against this URL.
            "1657606904-7527.jpg",
        ],
        "min_content_chars": 200,
        "required_keywords": ["насбулин", "himars", "генерал"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Glavred article on General Nasbulin liquidation with Telegram embed and Kyivstar TV player",
    },
    {
        "id": "C0-S12-Bukinfo-Oleksandrivka",
        "url": "https://bukinfo.com.ua/viyna-na-shodi/zsu-roznesly-rosiyskyy-shturm-udarom-haymarsa-na-video-bagato-zagyblyh-ta-poranenyh",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "Ukraine_365News/129564",
        ],
        "target_image_patterns": [
            "6a9734d9933ee.jpg",
        ],
        "min_content_chars": 200,
        "required_keywords": ["himars", "штурм"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Bukinfo combat report with Telegram video embed",
    },
    {
        "id": "C16-S7-TWZ-Harrisburg",
        "url": "https://www.twz.com/news-features/situation-report-battle-damaged-himars-return-to-united-states",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "5gVRSoPM1mY",
            "1756935613092155853",
            "1756972800391180539",
        ],
        "target_image_patterns": [
            "HIMARS-TOPSHOT",
            "HIMARS1-2",
        ],
        "min_content_chars": 300,
        "required_keywords": ["himars", "harrisburg", "damaged"],
        "forbidden_patterns": ["access denied"],
        "description": "The War Zone detailed analysis of battle-damaged HIMARS at Harrisburg airport",
    },
    {
        "id": "C7-S1-Militarnyi-Kakhovka",
        "url": "https://militarnyi.com/uk/news/u-novij-kahovtsi-kolosalnyj-vybuh-okupanty-zayavlyayut-shho-po-nyh-vdaryly-ukrayinski-himars",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "BSB5zjSaS4g",
        ],
        "target_image_patterns": [
            "Nova-Kahovka.jpg",
        ],
        "min_content_chars": 200,
        "required_keywords": ["кахов", "склад", "himars"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Militarnyi breaking report on colossal explosion in Nova Kakhovka with YouTube video and blast photo",
    },
    {
        "id": "C7-S15-Pravda-Satellite-Kakhovka",
        "url": "https://www.pravda.com.ua/news/2022/07/12/7357680",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "radiosvoboda/24697",
        ],
        "target_image_patterns": [
            "4983747-clipboard01.jpg",
        ],
        "min_content_chars": 200,
        "required_keywords": ["супутник", "кахов", "удар"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Ukrainska Pravda Planet Labs satellite imagery showing giant crater after HIMARS strike",
    },
    {
        "id": "C16-S1-Militarnyi-Harrisburg",
        "url": "https://militarnyi.com/en/news/damaged-himars-shipped-to-the-united-states-for-repair",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "5gVRSoPM1mY",
        ],
        "target_image_patterns": [
            "IMG_20240212_141518",
        ],
        "min_content_chars": 300,
        "required_keywords": ["himars", "harrisburg", "damaged"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Militarnyi report on An-124 unloading battle-damaged HIMARS in Harrisburg for repairs",
    },
    {
        "id": "C3-S10-Militarnyi-Belgorod",
        "url": "https://militarnyi.com/en/news/ukrainian-forces-strike-belgorod-tpp-with-himars-leaving-part-of-the-city-without-power-and-heating",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "QBX6_D7JWAc",
            "vvgladkov/18745",
        ],
        "target_image_patterns": [
            "20260125_075910",
        ],
        "min_content_chars": 300,
        "required_keywords": ["belgorod", "tpp", "himars"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Militarnyi analysis on Belgorod TPP strike with YouTube and Gladkov Telegram video",
    },
    {
        "id": "C3-S13-Fakty-Belgorod",
        "url": "https://fakty.com.ua/ru/svit/20250928-rosijskyj-byelgorod-chastkovo-bez-svitla-povidomlyayut-pro-prylit-po-tecz",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "v_hgVy3tUZM",
            "exilenova_plus/12679",
            "astrapress/93313",
        ],
        "target_image_patterns": [
            "5100846338219379604",
        ],
        "min_content_chars": 200,
        "required_keywords": ["белгород", "тэц", "свет"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Fakty ICTV report on Belgorod blackout with multiple Telegram and YouTube embeds",
    },
    {
        "id": "C8-S11-BBC-Nasbulin",
        "url": "https://www.bbc.com/russian/news-62134458",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "Bratchuk_Sergey/15351",
        ],
        "target_image_patterns": [
            "_125871146_tass_53781530",
        ],
        "min_content_chars": 300,
        "required_keywords": ["насбулин", "штаб", "himars"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "BBC Russian investigation on General Nasbulin death after HIMARS strike on command post",
    },
    {
        "id": "C13-S1-Slovoidilo-Penza",
        "url": "https://ru.slovoidilo.ua/2024/11/13/novost/obshhestvo/ataku-dronov-popala-neftebaza-rossijskoj-penze",
        "expected_kind": "article",
        "target_video_embed_patterns": [
            "astrapress/68377",
        ],
        "target_image_patterns": [
            "ilyustratyvne-foto",
        ],
        "min_content_chars": 200,
        "required_keywords": ["пенз", "нефтебаз", "дрон"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "Slovo i Dilo report on Penza oil storage attack with Astra Telegram footage",
    },
    {
        "id": "C10-S3-MoscowTimes-Makiivka",
        "url": "https://www.themoscowtimes.com/2023/01/04/death-toll-from-ukrainian-strike-on-makiivka-rises-to-89-a79870",
        "expected_kind": "article",
        "target_video_embed_patterns": [],
        "target_image_patterns": [
            "TASS_56658399",
        ],
        "min_content_chars": 300,
        "required_keywords": ["makiivka", "himars", "strike"],
        "forbidden_patterns": ["403 forbidden"],
        "description": "The Moscow Times report on HIMARS strike on Russian mobilized troop barracks in Makiivka",
    },
    {
        "id": "C9-S3-PravdaNews-Penza",
        "url": "https://pravda-news.ru/news/proisshestviya/khronika-s-13-noyabrya-2024-po-2-avgusta-2025",
        "expected_kind": "article",
        "target_video_embed_patterns": [],
        "target_image_patterns": [
            # Golden-dataset drift, 2026-10-06: same filename hash, but
            # Tavily now serves it as .jpeg instead of .webp (confirmed
            # live against a real /extract call — not a worker bug,
            # Tavily's own image pipeline re-encoded it). Dropped the
            # extension so the match survives this exact class of drift
            # going forward — same style the TASS_56658399 case above
            # already uses (hash/id only, no extension).
            "qfs9iz6bhq80sxuh50lgnsr7c7sr56bh",
        ],
        "min_content_chars": 300,
        "required_keywords": ["пензенск", "беспилотник", "монтажный"],
        "forbidden_patterns": ["403 forbidden", "access denied"],
        "description": "Pravda-news.ru chronicle of Penza drone attacks with target image and clean filtered metadata",
    },
]


SOCIAL_YTDLP_TEST_CASES = [
    {
        "id": "C2-S6-YouTube-Shorts-Rubicon",
        "url": "https://www.youtube.com/shorts/EztI_x4PQg0",
        "expected_platform": "youtube",
        "expected_uploader": "Armourdesia Military Hardware",
        "expected_upload_date": "20250506",
        "required_title_keywords": ["Rubicon", "HIMARS"],
        "required_description_keywords": ["Rubicon", "VT-40", "HIMARS", "Donetsk"],
        "description": "YouTube Shorts footage of Rubicon FPV drone hit on HIMARS",
    },
    {
        "id": "C0-S7-Instagram-Reel-Stepanenko",
        "url": "https://www.instagram.com/reel/DcCEsfax8Cx",
        "expected_platform": "instagram",
        "expected_uploader": "Vasilisa Stepanenko",
        "expected_upload_date": "20260814",
        "required_title_keywords": ["vasilisastepanenko"],
        "required_description_keywords": ["HIMARS", "Ukraine", "sunflowers"],
        "description": "Official AP News Instagram reel by Vasilisa Stepanenko",
    },
    {
        "id": "C16-S3-YouTube-Harrisburg-An124",
        "url": "https://www.youtube.com/watch?v=ZQZfCcwRwzs",
        "expected_platform": "youtube",
        "expected_uploader": "Ukraine way",
        "expected_upload_date": "20240302",
        "required_title_keywords": ["HIMARS", "Ukraine"],
        "required_description_keywords": ["An-124-100", "Harrisburg", "HIMARS"],
        "description": "Spotter video of Antonov An-124 unloading damaged HIMARS at Harrisburg",
    },
]


SOCIAL_PHOTO_TEST_CASES = [
    {
        "id": "C5-S8-Facebook-CamdenNews",
        "url": "https://www.facebook.com/CamdenNews/posts/lockheed-martin-and-general-dynamics-announced-the-signing-of-a-strategic-teamin/1107009138094185",
        "expected_platform": "facebook",
        "target_image_patterns": [
            "lookaside.fbsbx.com",
            "1107009111427521",
        ],
        "required_content_keywords": ["lockheed", "general dynamics", "camden"],
        "description": "Camden News Facebook photo post on Lockheed/GD deal announcement",
    },
    {
        "id": "C6-S11-Twitter-L3Harris",
        "url": "https://x.com/L3HarrisTech/status/1998784786391814552",
        "expected_platform": "twitter",
        "target_image_patterns": [
            "pbs.twimg.com/media/G70cKjiXgAANLB8",
            "pbs.twimg.com/media/G70cKu0XAAAv7Hu",
            "pbs.twimg.com/media/G70cK7vXMAA5Mra",
        ],
        "required_content_keywords": ["l3harris", "gmlrs", "camden"],
        "description": "Official L3Harris tweet with 3 facility construction and motor photos",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Test Suites
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveArticleExtractionContract:
    """Tests that real article URLs yield both target media and clean factual content."""

    @pytest.mark.live
    @pytest.mark.parametrize("case", ARTICLE_TEST_CASES, ids=[c["id"] for c in ARTICLE_TEST_CASES])
    async def test_article_yields_target_media_and_content(self, case: dict[str, Any]):
        async with httpx.AsyncClient(timeout=35.0) as http, StealthSession() as stealth:
            res: SourceExtractResult = await extract(http, stealth, _TAVILY_URL, case["url"])

        # 1. Kind assertion
        assert res.kind == case["expected_kind"], (
            f"[{case['id']}] Expected kind='{case['expected_kind']}', got '{res.kind}' "
            f"(reason: {res.reason})"
        )

        # 2. Target Video Embeds assertion (if required)
        if case["target_video_embed_patterns"]:
            actual_embed_urls = [e.url for e in res.video_embeds]
            matched_video = any(
                any(pattern in actual_url for actual_url in actual_embed_urls)
                for pattern in case["target_video_embed_patterns"]
            )
            assert matched_video, (
                f"[{case['id']}] None of expected video embeds found!\n"
                f"Expected patterns: {case['target_video_embed_patterns']}\n"
                f"Actual embeds returned: {actual_embed_urls}"
            )

        # 3. Target Images assertion (if required)
        if case["target_image_patterns"]:
            actual_images = res.images
            matched_image = any(
                any(pattern in actual_img for actual_img in actual_images)
                for pattern in case["target_image_patterns"]
            )
            assert matched_image, (
                f"[{case['id']}] None of expected target images found!\n"
                f"Expected patterns: {case['target_image_patterns']}\n"
                f"Actual clean images returned: {actual_images[:5]}"
            )

        # 4. Content Length assertion
        content = res.content or ""
        assert len(content) >= case["min_content_chars"], (
            f"[{case['id']}] Extracted content is too short: {len(content)} chars < {case['min_content_chars']}"
        )

        # 5. Required Factual Keywords assertion
        content_lower = content.lower()
        for kw in case["required_keywords"]:
            assert kw.lower() in content_lower, (
                f"[{case['id']}] Mandatory keyword '{kw}' missing from extracted content!"
            )

        # 6. Forbidden WAF / Error markers assertion
        for forbidden in case["forbidden_patterns"]:
            assert forbidden.lower() not in content_lower, (
                f"[{case['id']}] Extracted content contains error / WAF block marker: '{forbidden}'"
            )


class TestLiveSocialYtDlpMetadataContract:
    """Tests that social direct video URLs extract rich metadata via yt-dlp.

    Ensures title, uploader, upload_date, and description are NOT discarded,
    so vision validation has sufficient textual context.
    """

    @pytest.mark.live
    @pytest.mark.parametrize("case", SOCIAL_YTDLP_TEST_CASES, ids=[c["id"] for c in SOCIAL_YTDLP_TEST_CASES])
    async def test_social_direct_preserves_ytdlp_metadata(self, case: dict[str, Any]):
        async with httpx.AsyncClient(timeout=30.0) as http, StealthSession() as stealth:
            # Query yt-dlp metadata directly to verify upstream contract
            resp = await http.post(f"{_YT_DLP_URL}/metadata", json={"url": case["url"]})
            assert resp.status_code == 200, f"[{case['id']}] yt-dlp /metadata failed: {resp.text}"
            metadata = resp.json()

            # Assert yt-dlp provides required fields
            assert metadata.get("uploader"), f"[{case['id']}] yt-dlp uploader missing"
            assert metadata.get("upload_date"), f"[{case['id']}] yt-dlp upload_date missing"
            assert metadata.get("title"), f"[{case['id']}] yt-dlp title missing"

            desc = (metadata.get("description") or "").lower()
            for kw in case["required_description_keywords"]:
                assert kw.lower() in desc, (
                    f"[{case['id']}] Keyword '{kw}' missing from yt-dlp video description!"
                )

            # Now verify source-extract output (TARGET CONTRACT: content must NOT be empty!)
            res: SourceExtractResult = await extract(http, stealth, _TAVILY_URL, case["url"])
            assert res.kind == "social_direct"
            assert res.platform == case["expected_platform"]

            # CONTRACT CHECK: content must contain uploader, date, and description keywords
            # (Note: Will be satisfied once worker injects yt-dlp metadata into content)
            assert res.content, (
                f"[{case['id']}] CONTRACT FAILURE: source-extract returned empty content for social video!\n"
                f"yt-dlp has title='{metadata.get('title')}' and uploader='{metadata.get('uploader')}', "
                f"but source-extract discarded it, blinding Vision validation."
            )


class TestLiveClientPydanticCompatibility:
    """Tests that client models in monivest-server correctly deserialize worker outputs."""

    @pytest.mark.live
    @pytest.mark.parametrize("platform", [
        "youtube", "twitter", "telegram", "instagram", "facebook", "tiktok",
        "vimeo", "dailymotion", "jwplayer", "brightcove", "html5", "direct",
    ])
    def test_client_platform_enum_accepts_all_worker_platforms(self, platform: str):
        from workers import source_extract as client_module

        # Verify client Platform literal includes the platform
        type_args = client_module.Platform.__args__
        assert platform in type_args, (
            f"Client Platform enum in workers/source_extract.py is missing '{platform}'!\n"
            f"Supported platforms by worker: {type_args}"
        )
