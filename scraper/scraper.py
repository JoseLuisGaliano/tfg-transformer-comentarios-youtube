import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_KEY = os.getenv("YOUTUBE_API_KEY")

def build_client():
    if not API_KEY:
        sys.exit("ERROR: Falta la variable de entorno YOUTUBE_API_KEY.")
    return build("youtube", "v3", developerKey=API_KEY)


# búsqueda amplia vía search.list (paginable)
def search_video_ids_via_search(
    yt,
    query: str,
    max_results_total: int,
    region_code: Optional[str],
    language_filter: Optional[str] = None,
    order: str = "relevance",
) -> List[str]:

    video_ids: List[str] = []
    seen: Set[str] = set()
    next_page_token = None

    while len(video_ids) < max_results_total:
        to_fetch = min(50, max_results_total - len(video_ids))  # search.list máx 50
        req_kwargs = dict(
            part="id",
            type="video",
            q=query,
            maxResults=to_fetch,
            order=order,  # 'date' | 'rating' | 'relevance' | 'title' | 'videoCount' | 'viewCount'
            pageToken=next_page_token,
        )
        if region_code:
            req_kwargs["regionCode"] = region_code
        if language_filter:
            req_kwargs["relevanceLanguage"] = language_filter

        res = yt.search().list(**req_kwargs).execute()

        items = res.get("items", [])
        for it in items:
            id_info = it.get("id", {}) or {}
            vid = id_info.get("videoId")
            if not vid:
                continue
            if vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
                if len(video_ids) >= max_results_total:
                    break

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


# detalles completos vía videos.list (en lotes)
def fetch_videos_metadata_by_ids(
    yt,
    video_ids: List[str],
) -> List[Dict[str, Any]]:

    videos: List[Dict[str, Any]] = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        if not batch:
            continue
        res = yt.videos().list(
            part="snippet,topicDetails",
            id=",".join(batch),
            maxResults=len(batch),
        ).execute()
        for item in res.get("items", []):
            vid = item.get("id")
            sn = item.get("snippet", {}) or {}
            td = item.get("topicDetails", {}) or {}
            ch = sn.get("channelId")
            if vid and ch:
                videos.append({
                    "video_id": vid,
                    "channel_id": ch,
                    "video_title": sn.get("title", "") or "",
                    "channel_title": sn.get("channelTitle", "") or "",
                    "description": sn.get("description", "") or "",
                    "language": (sn.get("defaultLanguage") or "").lower(),
                    "category": sn.get("categoryId", "") or "",
                    "topicCategories": td.get("topicCategories", []) or [],
                })
        # pequeña pausa para la API
        time.sleep(0.05)
    return videos


def search_videos_combined(
    yt,
    query: str,
    max_results_total: int,
    region_code: Optional[str],
    language_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Flujo:
      1) search.list -> muchos video_ids (paginable)
      2) videos.list -> detalles completos (snippet + topicDetails)
      3) filtro final por defaultLanguage (estricto)
      4) deduplicación y recorte a max_results_total
    """
    # 1) IDs amplios por search
    ids = search_video_ids_via_search(
        yt=yt,
        query=query,
        max_results_total=max_results_total,
        region_code=region_code,
        language_filter=language_filter,  # ayuda a priorizar resultados en ese idioma
        order="relevance",
    )
    print(f"IDs recuperados con search: {len(ids)}")

    if not ids:
        return []

    # 2) Detalles completos por videos.list
    videos = fetch_videos_metadata_by_ids(yt, ids)

    '''
    # 3) Filtro estricto por defaultLanguage
    if language_filter:
        lang = language_filter.lower()
        videos = [v for v in videos if (v.get("language") or "").lower() == lang]
        print(f"Vídeos tras filtro estricto de idioma (defaultLanguage={lang}): {len(videos)}")
    '''
    
    # 4) Deduplicación y recorte final
    seen = set()
    unique: List[Dict[str, Any]] = []
    for v in videos:
        vid = v["video_id"]
        if vid not in seen:
            seen.add(vid)
            unique.append(v)
        if len(unique) >= max_results_total:
            break



def fetch_top_comments(yt, video_id: str, max_comments: int) -> List[Dict[str, Any]]:

    if max_comments <= 0:
        return []
    try:
        req = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            textFormat="plainText",
            order="relevance",
        )
        res = req.execute()
    except HttpError:
        # comentarios deshabilitados u otro error
        return []

    comments: List[Dict[str, Any]] = []
    for item in res.get("items", []):
        top = item.get("snippet", {}).get("topLevelComment", {}) or {}
        sn = top.get("snippet", {}) or {}
        text = sn.get("textOriginal") or sn.get("textDisplay") or ""
        if not text:
            continue
        author_ch = sn.get("authorChannelId") or {}
        author_id = author_ch.get("value", "") or ""
        author_name = sn.get("authorDisplayName", "") or ""
        likes = sn.get("likeCount", 0) or 0
        comments.append({
            "text": text,
            "author_id": author_id,
            "author_name": author_name,
            "likes": likes,
        })
        if len(comments) >= max_comments:
            break
    return comments


def ensure_csv_with_header(path: Path):

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "text",
                "author_id",
                "author_name",
                "likes",
                "video_id",
                "video_title",
                "channel_id",
                "channel_title",
                "description",
                "language",
                "category",
                "topicCategories",  # string con ' | ' si hay múltiples
                "label",
            ])


def load_existing_video_ids(path: Path) -> Set[str]:

    existing: Set[str] = set()
    if not path.exists():
        return existing
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        vid_idx = None
        if header:
            try:
                vid_idx = header.index("video_id")
            except ValueError:
                vid_idx = 4  # en el header actual, video_id es índice 4
        else:
            vid_idx = 4
        for row in reader:
            if not row or len(row) <= vid_idx:
                continue
            existing.add(row[vid_idx])
    return existing

def append_rows(path: Path, rows: List[List[Any]]):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# convierte topicCategories (lista) a un único string
def serialize_topics(value: Any, sep: str = " | ") -> str:

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        seen = set()
        cleaned = []
        for x in value:
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                cleaned.append(str(x))
        return sep.join(cleaned)
    except TypeError:
        return str(value)


def build_rows_for_video(yt, video: Dict[str, Any], max_comments: int, query: str) -> List[List[Any]]:

    rows: List[List[Any]] = []
    comments = fetch_top_comments(yt, video["video_id"], max_comments=max_comments)
    topics_str = serialize_topics(video.get("topicCategories", []), sep=" | ")
    for c in comments:
        rows.append([
            c.get("text", ""),
            c.get("author_id", ""),
            c.get("author_name", ""),
            c.get("likes", 0),
            video.get("video_id", ""),
            video.get("video_title", ""),
            video.get("channel_id", ""),
            video.get("channel_title", ""),
            video.get("description", ""),
            video.get("language", ""),
            video.get("category", ""),
            topics_str,
            query,
        ])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Scraper de comentarios en YouTube por queries (keywords)")
    parser.add_argument("query", type=str, help="query que pasamos a search.list (temática)")
    parser.add_argument("num_videos", type=int, help="Cantidad de vídeos a recuperar")
    parser.add_argument("--output", "-o", default="dataset.csv", help="Fichero CSV de salida")
    parser.add_argument("--region", "-r", default="ES", help="regionCode opcional")
    parser.add_argument("--language", "-l", default="es", help="Filtrar por lenguaje")
    parser.add_argument("--comments", type=int, default=100, help="Máximo de comentarios por vídeo")
    args = parser.parse_args()

    yt = build_client()

    out_path = Path(args.output)
    ensure_csv_with_header(out_path)

    # 1) Recuperar vídeos con combinación search + videos
    videos = search_videos_combined(
        yt=yt,
        query=args.query,
        max_results_total=max(0, args.num_videos),
        region_code=args.region,
        language_filter=args.language,
    )
    print(f"Vídeos recuperados: {len(videos)}")

    if not videos:
        print("No se encontraron vídeos para la categoría/idioma indicada.")
        return

    # 2) Cargar IDs ya presentes y filtrar
    existing_ids = load_existing_video_ids(out_path)
    videos_to_use = [v for v in videos if v["video_id"] not in existing_ids]

    total_rows = 0
    for v in videos_to_use:
        rows = build_rows_for_video(yt, v, max_comments=args.comments, query = args.query)
        if rows:
            append_rows(out_path, rows)
            total_rows += len(rows)
        time.sleep(0.1)  # pequeña pausa para ser amable con la API

    # Cuántos de los recuperados se han usado realmente
    print(f"Vídeos realmente usados (no repetidos): {len(videos_to_use)}")

    # Resumen final
    print(f"Listo. Vídeos procesados: {len(videos_to_use)}. Filas añadidas (comentarios): {total_rows}. Archivo: {out_path}")

if __name__ == "__main__":
    main()

