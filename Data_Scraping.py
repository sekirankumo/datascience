import os
import random
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Genius API 基础类
# -------------------------------
class GeniusAPI:
    def __init__(self, access_token: str = None):
        self.base_url = "https://api.genius.com"
        self.access_token = access_token or os.getenv('GENIUS_ACCESS_TOKEN')
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def _make_request(self, endpoint: str, method: str = 'GET',
                     params: Dict = None, data: Dict = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, params=params)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"错误响应: {e.response.text}")
            return {}

# -------------------------------
# Genius 客户端
# -------------------------------
class GeniusClient(GeniusAPI):
    def search(self, query: str, per_page: int = 10, page: int = 1) -> Dict[str, Any]:
        endpoint = "/search"
        params = {'q': query, 'per_page': per_page, 'page': page}
        return self._make_request(endpoint, params=params)

    def get_song(self, song_id: int, text_format: str = "plain") -> Dict[str, Any]:
        endpoint = f"/songs/{song_id}"
        params = {'text_format': text_format}
        return self._make_request(endpoint, params=params)

    def get_artist(self, artist_id: int, text_format: str = "plain") -> Dict[str, Any]:
        endpoint = f"/artists/{artist_id}"
        params = {'text_format': text_format}
        return self._make_request(endpoint, params=params)

    def get_artist_songs(self, artist_id: int, sort: str = "title",
                        per_page: int = 20, page: int = 1) -> Dict[str, Any]:
        endpoint = f"/artists/{artist_id}/songs"
        params = {'sort': sort, 'per_page': per_page, 'page': page}
        return self._make_request(endpoint, params=params)

    def get_annotation(self, annotation_id: int, text_format: str = "plain") -> Dict[str, Any]:
        endpoint = f"/annotations/{annotation_id}"
        params = {'text_format': text_format}
        return self._make_request(endpoint, params=params)

    def get_referents(self, web_page_id: int = None, song_id: int = None,
                     created_by_id: int = None, per_page: int = 20, page: int = 1) -> Dict[str, Any]:
        endpoint = "/referents"
        params = {'per_page': per_page, 'page': page}
        if web_page_id: params['web_page_id'] = web_page_id
        elif song_id: params['song_id'] = song_id
        elif created_by_id: params['created_by_id'] = created_by_id
        return self._make_request(endpoint, params=params)

# -------------------------------
# 自动抓取所有分页数据
# -------------------------------
def fetch_all_pages(fetch_func, **kwargs) -> List[Dict]:
    all_items = []
    page = 1
    while True:
        try:
            data = fetch_func(page=page, **kwargs)
        except Exception as e:
            print(f"⚠️ 第 {page} 页请求失败: {e}")
            break

        if not data or "response" not in data:
            break

        if "hits" in data["response"]:
            items = data["response"]["hits"]
        elif "songs" in data["response"]:
            items = data["response"]["songs"]
        elif "referents" in data["response"]:
            items = data["response"]["referents"]
        else:
            break

        if not items:
            break

        all_items.extend(items)
        page += 1

    return all_items

# -------------------------------
# 转换为表格行
# -------------------------------
def song_to_row(song):
    return {
        "id": song.get("id"),
        "title": song.get("title"),
        "url": song.get("url"),
        "artist": song.get("primary_artist", {}).get("name"),
        "artist_id": song.get("primary_artist", {}).get("id")
    }

def referent_to_row(r):
    return {
        "referent_id": r.get("id"),
        "song_id": r.get("song_id"),
        "annotation_id": r.get("annotation", {}).get("id"),
        "fragment": r.get("fragment"),
    }

def annotation_to_row(a):
    anno = a.get("response", {}).get("annotation", {})
    body = anno.get("body", {}).get("plain")
    return {
        "annotation_id": anno.get("id"),
        "author_id": anno.get("author_id"),
        "votes_total": anno.get("votes_total"),
        "body": body
    }

# -------------------------------
# 随机抽歌并抓取信息
# -------------------------------
def random_pick_songs(client, keyword: str, k: int = 5):
    print(f"正在搜索歌曲池：{keyword} ...")
    all_hits = fetch_all_pages(client.search, query=keyword, per_page=50)

    if not all_hits:
        print("无搜索结果")
        return None

    song_pool = [h["result"] for h in all_hits]
    print(f"共找到 {len(song_pool)} 首歌")

    chosen_songs = random.sample(song_pool, min(k, len(song_pool)))

    df_basic, df_song_details, df_artist, df_referents, df_annotations = [], [], [], [], []

    for song in chosen_songs:
        song_id = song.get("id")
        artist_id = song.get("primary_artist", {}).get("id")

        # 基本信息
        df_basic.append(song_to_row(song))

        # 歌曲详细信息
        try:
            song_detail = client.get_song(song_id, text_format="plain")
            if song_detail and "response" in song_detail and "song" in song_detail["response"]:
                df_song_details.append(song_detail["response"]["song"])
        except Exception as e:
            print(f"⚠️ 歌曲 {song_id} 请求失败: {e}")

        # 艺术家信息
        try:
            artist_detail = client.get_artist(artist_id)
            if artist_detail and "response" in artist_detail and "artist" in artist_detail["response"]:
                df_artist.append(artist_detail["response"]["artist"])
        except Exception as e:
            print(f"⚠️ 艺术家 {artist_id} 请求失败: {e}")

        # referents
        referents_raw = fetch_all_pages(client.get_referents, song_id=song_id)
        for r in referents_raw:
            df_referents.append(referent_to_row(r))
            anno_id = r.get("annotation", {}).get("id")
            if anno_id:
                try:
                    detail = client.get_annotation(anno_id)
                    df_annotations.append(annotation_to_row(detail))
                except Exception as e:
                    print(f"⚠️ 注释 {anno_id} 请求失败: {e}")

    tables = {
        "basic_info": pd.DataFrame(df_basic),
        "song_details": pd.json_normalize(df_song_details),
        "artist_info": pd.json_normalize(df_artist),
        "referents": pd.DataFrame(df_referents),
        "annotations": pd.DataFrame(df_annotations),
    }

    return tables

# -------------------------------
# 合并表格
# -------------------------------
def merge_all_tables(tables):
    df_basic = tables.get("basic_info", pd.DataFrame())
    df_song = tables.get("song_details", pd.DataFrame())
    df_artist = tables.get("artist_info", pd.DataFrame())
    df_ref = tables.get("referents", pd.DataFrame())
    df_ann = tables.get("annotations", pd.DataFrame())

    merged = df_basic.copy()

    if not df_song.empty:
        merged = merged.merge(df_song, how="left", left_on="id", right_on="id", suffixes=("", "_song"))
    if not df_artist.empty:
        merged = merged.merge(df_artist, how="left", left_on="artist_id", right_on="id", suffixes=("", "_artist"))
    if not df_ref.empty:
        merged = merged.merge(df_ref, how="left", left_on="id", right_on="song_id", suffixes=("", "_ref"))
    if not df_ann.empty and "annotation_id" in df_ann.columns:
        merged = merged.merge(df_ann, how="left", left_on="annotation_id", right_on="annotation_id", suffixes=("", "_ann"))

    return merged

# -------------------------------
# 使用示例
# -------------------------------
if __name__ == "__main__":
    GENIUS_ACCESS_TOKEN="dziQFFQsKiT0EvULO9ZOWV8v--v4dMLRlHrLoeRB3zfLvIfao-YmZj89-fxTQ8Lm"
    client = GeniusClient(access_token=GENIUS_ACCESS_TOKEN)

    # 抓取随机歌曲信息
    tables = random_pick_songs(client, keyword="love", k=200)
    if tables:
        final_table = merge_all_tables(tables)
        print(final_table.head())

        # 可选：导出 Excel
        final_table.to_excel("genius_songs.xlsx", index=False)
