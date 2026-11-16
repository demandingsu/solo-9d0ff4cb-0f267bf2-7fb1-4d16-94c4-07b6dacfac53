"""人工复核挑选：求解逻辑、候选范围、版本固定、失败回滚与页面。"""
import csv
import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.grouping import assign_split
from app.main import Settings, create_app
from app.review import CandidateGroup, Infeasible, TooManyCandidates, select_groups


def cg(gid, species, started="2026-03-01T08:00:00Z"):
    return CandidateGroup(gid, started, frozenset(species), 1)


# ---------------------------------------------------------------- 求解器纯逻辑

def test_single_group_covers_multiple_quotas():
    cands = [cg("P1", {"boar"}), cg("P2", {"deer"}), cg("P3", {"boar", "deer"})]
    assert select_groups(cands, {"boar": 1, "deer": 1}, 10) == ["P3"]


def test_minimal_count_and_stable_tiebreak():
    cands = [cg("P1", {"boar"}), cg("P3", {"boar"}), cg("P2", {"deer"})]
    # boar 可由 P1 或 P3 抵，并列取组号较小者
    assert select_groups(cands, {"boar": 1}, 10) == ["P1"]
    # 打乱输入顺序结果不变（稳定）
    assert select_groups(list(reversed(cands)), {"boar": 1}, 10) == ["P1"]


def test_tiebreak_prefers_lexicographically_smaller_combo():
    cands = [cg("P1", {"boar"}), cg("P2", {"deer"}), cg("P3", {"boar"}), cg("P4", {"deer"})]
    # 需 2 组；可行组合 {P1,P2} {P1,P4} {P2,P3} {P3,P4}，取字典序最小
    assert select_groups(cands, {"boar": 1, "deer": 1}, 10) == ["P1", "P2"]


def test_quota_counts_groups_not_photos():
    # 同组同物种多张照片仍只算一组：P1 组里 3 张 boar 也只抵 1 组配额
    cands = [CandidateGroup("P1", "2026-03-01T08:00:00Z", frozenset({"boar"}), 3)]
    with pytest.raises(Infeasible) as ei:
        select_groups(cands, {"boar": 2}, 10)
    assert ei.value.reason == "shortage"
    assert "可用组不足" in ei.value.detail


def test_shortage_when_species_cover_too_small():
    cands = [cg("P1", {"boar"}), cg("P2", {"boar"})]
    with pytest.raises(Infeasible) as ei:
        select_groups(cands, {"boar": 3}, 10)
    assert ei.value.reason == "shortage"
    assert "可用组不足" in ei.value.detail


def test_cap_when_min_solution_exceeds_max_groups():
    cands = [cg("P1", {"boar"}), cg("P2", {"deer"}), cg("P3", {"fox"})]
    with pytest.raises(Infeasible) as ei:
        select_groups(cands, {"boar": 1, "deer": 1, "fox": 1}, 2)
    assert ei.value.reason == "cap"
    assert "总上限不足" in ei.value.detail
    assert "至少需 3 组" in ei.value.detail


def test_too_many_candidates():
    cands = [cg(f"P{i:02d}", {"boar"}) for i in range(19)]
    with pytest.raises(TooManyCandidates) as ei:
        select_groups(cands, {"boar": 1}, 10)
    assert ei.value.count == 19
    assert ei.value.limit == 18


def test_empty_quota_rejected():
    with pytest.raises(ValueError):
        select_groups([cg("P1", {"boar"})], {"boar": 0}, 10)


# ---------------------------------------------------------------- 端到端

def make_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["photo_no", "camera_no", "captured_at", "species", "rel_path"])
    w.writerows(rows)
    return buf.getvalue().encode()


# CAM01 2026-03-01：组 P001{boar}、组 P002{deer}、组 P003{boar,deer}（P003/P004 同组）
BASE = make_csv([
    ["P001", "CAM01", "2026-03-01T08:00:00Z", "wild_boar", "cam01/20260301/P001.jpg"],
    ["P002", "CAM01", "2026-03-01T09:00:00Z", "roe_deer", "cam01/20260301/P002.jpg"],
    ["P003", "CAM01", "2026-03-01T10:00:00Z", "wild_boar", "cam01/20260301/P003.jpg"],
    ["P004", "CAM01", "2026-03-01T10:00:30Z", "roe_deer", "cam01/20260301/P004.jpg"],
    ["P005", "CAM01", "2026-03-02T08:00:00Z", "wild_boar", "cam01/20260302/P005.jpg"],  # 另一天
    ["P006", "CAM02", "2026-03-01T08:00:00Z", "wild_boar", "cam02/20260301/P006.jpg"],  # 另一相机
])


@pytest.fixture()
def env(tmp_path):
    # val_percent=0：全部组进训练索引，候选 = 全部组
    settings = Settings(data_dir=tmp_path / "data", val_percent=0, gap_seconds=60)
    app = create_app(settings)
    client = TestClient(app, follow_redirects=True)
    return client, settings


def upload(client, data, name="b.csv"):
    return client.post("/upload", files={"file": (name, data, "text/csv")})


def create_list(client, vid, camera, date, quotas, max_groups=10):
    data = {"version_id": str(vid), "camera_no": camera, "group_date": date, "max_groups": str(max_groups)}
    data.update({f"quota_{s}": str(n) for s, n in quotas.items()})
    return client.post("/review", data=data)


def table_count(settings, table):
    conn = sqlite3.connect(settings.db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def list_groups(settings, lid):
    conn = sqlite3.connect(settings.db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT group_id FROM review_list_groups WHERE list_id=? ORDER BY group_id", (lid,))]
    finally:
        conn.close()


def list_photos(settings, lid):
    conn = sqlite3.connect(settings.db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT photo_no FROM review_list_photos WHERE list_id=? ORDER BY photo_no", (lid,))]
    finally:
        conn.close()


def test_multi_species_group_fills_multiple_quotas(env):
    client, settings = env
    upload(client, BASE)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1, "roe_deer": 1})
    assert r.status_code == 200
    assert "复核清单 #1" in r.text
    # P003 组同时含两物种，1 组即满足两个配额；该组全部照片入清单
    assert list_groups(settings, 1) == ["P003"]
    assert list_photos(settings, 1) == ["P003", "P004"]
    # 页面列出原图相对路径
    assert "cam01/20260301/P003.jpg" in r.text
    assert "cam01/20260301/P004.jpg" in r.text


def test_minimal_and_stable_tiebreak_end_to_end(env):
    client, settings = env
    upload(client, BASE)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert "复核清单 #1" in r.text
    assert list_groups(settings, 1) == ["P001"]  # P001/P003 并列，取组号小者
    # 再次提交结果一致（稳定选取）
    r2 = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert "复核清单 #2" in r2.text
    assert list_groups(settings, 2) == ["P001"]


def test_camera_and_date_scope(env):
    client, settings = env
    upload(client, BASE)
    # 03-02 只有 P005 一组 boar
    r = create_list(client, 1, "CAM01", "2026-03-02", {"wild_boar": 1})
    assert list_groups(settings, 1) == ["P005"]
    # CAM02 在 03-01 有 P006
    r = create_list(client, 1, "CAM02", "2026-03-01", {"wild_boar": 1})
    assert list_groups(settings, 2) == ["P006"]
    # CAM02 在 03-02 无候选 -> 可用组不足
    r = create_list(client, 1, "CAM02", "2026-03-02", {"wild_boar": 1})
    assert "可用组不足" in r.text
    assert table_count(settings, "review_lists") == 2


def test_shortage_reports_and_creates_nothing(env):
    client, settings = env
    upload(client, BASE)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 5})
    assert "可用组不足" in r.text
    assert "wild_boar" in r.text
    assert table_count(settings, "review_lists") == 0
    assert table_count(settings, "review_list_groups") == 0
    assert table_count(settings, "review_list_photos") == 0


def test_cap_reports_and_creates_nothing(env):
    client, settings = env
    upload(client, BASE)
    # boar 需 2 组（P001、P003），上限 1 -> 总上限不足
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 2}, max_groups=1)
    assert "总上限不足" in r.text
    assert "至少需 2 组" in r.text
    assert table_count(settings, "review_lists") == 0


def test_too_many_candidates_asks_to_narrow(env):
    client, settings = env
    rows = []
    for i in range(19):  # 19 个独立组，相邻间隔 2 分钟（> 60s 阈值）
        minute = (i * 2) % 60
        hour = 8 + (i * 2) // 60
        rows.append([f"P{i:03d}", "CAM01", f"2026-03-01T{hour:02d}:{minute:02d}:00Z", "wild_boar", f"cam01/P{i:03d}.jpg"])
    upload(client, make_csv(rows))
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert "超过 18 组上限" in r.text
    assert "缩小范围" in r.text
    assert table_count(settings, "review_lists") == 0


def test_only_train_index_groups_are_candidates(tmp_path):
    # 默认 20% 验证比例：找一个落 val 的组号和一个落 train 的组号
    salt = "wildlife-trap-v1"
    val_no = next(f"P{i:04d}" for i in range(1, 999) if assign_split(f"P{i:04d}", 20, salt) == "val")
    train_no = next(f"P{i:04d}" for i in range(1, 999) if assign_split(f"P{i:04d}", 20, salt) == "train")
    settings = Settings(data_dir=tmp_path / "data", val_percent=20, gap_seconds=60)
    client = TestClient(create_app(settings), follow_redirects=True)
    upload(client, make_csv([
        [train_no, "CAM01", "2026-03-01T08:00:00Z", "wild_boar", f"cam01/{train_no}.jpg"],
        [val_no, "CAM01", "2026-03-01T09:00:00Z", "roe_deer", f"cam01/{val_no}.jpg"],
    ]))
    # val 组的物种在候选中不存在 -> 可用组不足
    r = create_list(client, 1, "CAM01", "2026-03-01", {"roe_deer": 1})
    assert "可用组不足" in r.text
    # train 组可选
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert list_groups(settings, 1) == [train_no]


def test_save_failure_rolls_back_and_old_lists_survive(env, monkeypatch):
    client, settings = env
    upload(client, BASE)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"roe_deer": 1})
    assert "复核清单 #1" in r.text  # 先有一份好清单

    import app.main as main

    def boom(*args, **kwargs):
        raise RuntimeError("磁盘写入失败（模拟）")

    monkeypatch.setattr(main, "_insert_review_photos", boom)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert "保存失败" in r.text
    # 不留半份清单：三张表都没有新清单的痕迹
    assert table_count(settings, "review_lists") == 1
    assert table_count(settings, "review_list_groups") == 1
    assert table_count(settings, "review_list_photos") == 1
    # 旧清单仍可查看
    r = client.get("/review/1")
    assert r.status_code == 200
    assert "复核清单 #1" in r.text

    monkeypatch.undo()
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    assert "复核清单 #2" in r.text  # 恢复后可正常生成


def test_list_pinned_to_version_despite_later_publish(env):
    client, settings = env
    # v1：CAM01 两组 boar（间隔 120 秒）
    upload(client, make_csv([
        ["P001", "CAM01", "2026-03-01T08:00:00Z", "wild_boar", "cam01/P001.jpg"],
        ["P002", "CAM01", "2026-03-01T08:02:00Z", "wild_boar", "cam01/P002.jpg"],
    ]))
    # v2：补传桥接照片，两组合并为一组
    upload(client, make_csv([
        ["P003", "CAM01", "2026-03-01T08:01:00Z", "wild_boar", "cam01/P003.jpg"],
    ]))
    assert table_count(settings, "versions") == 2
    # 对 v1 挑选：v1 有两组 boar，配额 2 组可满足；若错用 v2 索引（合并后仅 1 组）必报不足
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 2})
    assert "复核清单 #1" in r.text
    assert list_groups(settings, 1) == ["P001", "P002"]
    assert list_photos(settings, 1) == ["P001", "P002"]
    # 清单固定 v1
    conn = sqlite3.connect(settings.db_path)
    vid = conn.execute("SELECT version_id FROM review_lists WHERE id=1").fetchone()[0]
    conn.close()
    assert vid == 1
    # 对 v2 同一配额则可用组不足（合并后只 1 组）
    r = create_list(client, 2, "CAM01", "2026-03-01", {"wild_boar": 2})
    assert "可用组不足" in r.text


def test_old_list_viewable_after_new_publish(env):
    client, settings = env
    upload(client, BASE)
    create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1})
    before = client.get("/review/1").text
    # 发布 v2（新照片），清单内容不变
    upload(client, make_csv([
        ["P100", "CAM09", "2026-03-03T10:00:00Z", "red_fox", "cam09/P100.jpg"],
    ]))
    after = client.get("/review/1").text
    assert after == before
    assert "复核清单 #1" in after


def test_review_create_interleaved_with_publish_stays_consistent(env):
    """补传发布与清单生成交错：清单完整落库，且候选始终来自所选版本的索引。"""
    from concurrent.futures import ThreadPoolExecutor

    client, settings = env
    upload(client, BASE)  # v1
    new_batch = make_csv([
        ["P200", "CAM09", "2026-03-03T10:00:00Z", "red_fox", "cam09/P200.jpg"],
    ])
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(create_list, client, 1, "CAM01", "2026-03-01", {"wild_boar": 1, "roe_deer": 1})
        f2 = pool.submit(upload, client, new_batch, "concurrent.csv")
        r1, r2 = f1.result(), f2.result()
    assert r1.status_code == 200 and r2.status_code == 200
    # 清单完整：头、组、照片三表一致（不混用 v2 索引，也不留半份）
    assert table_count(settings, "review_lists") == 1
    assert list_groups(settings, 1) == ["P003"]
    assert list_photos(settings, 1) == ["P003", "P004"]
    conn = sqlite3.connect(settings.db_path)
    vid = conn.execute("SELECT version_id FROM review_lists WHERE id=1").fetchone()[0]
    conn.close()
    assert vid == 1
    # 并发上传也正常发布
    assert table_count(settings, "versions") == 2


def test_form_page_shows_version_camera_species(env):
    client, settings = env
    upload(client, BASE)
    r = client.get("/review/new?version_id=1")
    assert r.status_code == 200
    assert "新建人工复核挑选" in r.text
    assert 'name="camera_no"' in r.text
    assert "CAM01" in r.text and "CAM02" in r.text
    assert 'name="quota_wild_boar"' in r.text
    assert 'name="quota_roe_deer"' in r.text
    assert 'name="max_groups"' in r.text
    # 首页有入口
    assert "/review/new" in client.get("/").text


def test_zero_quota_rejected(env):
    client, settings = env
    upload(client, BASE)
    r = create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 0})
    assert "请至少为一个物种" in r.text
    assert table_count(settings, "review_lists") == 0


def test_review_index_and_photos_csv(env):
    client, settings = env
    upload(client, BASE)
    create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1, "roe_deer": 1})
    r = client.get("/review")
    assert "复核清单" in r.text and "#1" in r.text
    r = client.get("/review/1/photos.csv")
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert [row["photo_no"] for row in rows] == ["P003", "P004"]
    assert rows[0]["rel_path"] == "cam01/20260301/P003.jpg"
    assert rows[0]["group_id"] == "P003"


def test_quotas_saved_as_json_and_shown(env):
    client, settings = env
    upload(client, BASE)
    create_list(client, 1, "CAM01", "2026-03-01", {"wild_boar": 1, "roe_deer": 1})
    conn = sqlite3.connect(settings.db_path)
    quotas = json.loads(conn.execute("SELECT quotas FROM review_lists WHERE id=1").fetchone()[0])
    conn.close()
    assert quotas == {"roe_deer": 1, "wild_boar": 1}
    r = client.get("/review/1")
    assert "wild_boar ≥ 1 组" in r.text
