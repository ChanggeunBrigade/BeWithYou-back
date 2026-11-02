import datetime
from collections.abc import Iterator

import psycopg2
import psycopg2.extensions
from psycopg2 import sql

from settings import DB_TIMEZONE, SingletonMeta, config

FETCH_SIZE = 10_000  # 서버 쪽 커서에서 한 번에 가져오는 행 수


def time_filter(start, end) -> tuple[sql.Composable, dict]:
    """
    `time` 컬럼 기간 조건. start/end는 DB_TIMEZONE 기준 시각(문자열 또는 datetime)이며
    None이면 제한하지 않는다. [start, end) 구간.
    """
    conditions, params = [sql.SQL("true")], {}
    if start is not None:
        conditions.append(sql.SQL("time >= %(start)s"))
        params["start"] = start
    if end is not None:
        conditions.append(sql.SQL("time < %(end)s"))
        params["end"] = end
    return sql.SQL(" and ").join(conditions), params


class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.conn: psycopg2.extensions.connection = psycopg2.connect(
            host=config["PSQL_HOST"],
            port=config["PSQL_PORT"],
            database="data",
            user=config["PSQL_USER"],
            password=config["PSQL_PASS"],
        )
        self.migrate()

    def migrate(self):
        """
        예전에 만든 DB에도 label.ts(epoch 초) 컬럼을 추가하고, 비어 있으면 라벨을 붙인
        opencv 프레임의 ts로 채운다. 라벨 시각을 `time` 컬럼(DB 시간대에 따라 해석이
        달라짐) 대신 신호와 같은 epoch 기준으로 맞추기 위함이다.
        """
        with self.conn.cursor() as cursor:
            cursor.execute("alter table label add column if not exists ts double precision")
            cursor.execute(
                "update label l set ts = (o.data->>'ts')::float8 from opencv o"
                " where l.ts is null and o.time = l.time and o.data ? 'ts'"
            )
        self.conn.commit()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def _iter(self, query: sql.Composable, params: dict) -> Iterator[tuple]:
        # 이름 있는 커서(서버 쪽 커서)로 FETCH_SIZE씩 나눠 받아 테이블 전체를 메모리에 올리지 않는다
        with self.conn.cursor(name="bewithyou_stream") as cursor:
            cursor.itersize = FETCH_SIZE
            cursor.execute(query, params)
            yield from cursor
        self.conn.commit()

    def get_audio_power(self, step: float, start=None, end=None) -> list[tuple[int, float]]:
        """
        오디오 평균 전력을 step초 격자로 DB에서 미리 집계한다.

        예전 형식(샘플 하나씩 `data`, 초당 44100행)도 격자당 한 행으로 줄어든다.
        :return: (격자 번호 = floor(ts / step), 평균 전력) 목록
        """
        where, params = time_filter(start, end)
        query = sql.SQL(
            "select floor((data->>'ts')::float8 / %(step)s)::bigint as bucket,"
            " avg(power(coalesce((data->>'rms')::float8, (data->>'data')::float8), 2))"
            " from audio where {} and data ? 'ts' and (data ? 'rms' or data ? 'data')"
            " group by bucket order by bucket"
        ).format(where)
        return list(self._iter(query, {**params, "step": step}))

    def iter_csi(self, start=None, end=None) -> Iterator[tuple[float, list, list]]:
        """(ts, amplitudes, phases)를 시간 순서로 스트리밍한다."""
        where, params = time_filter(start, end)
        query = sql.SQL(
            "select (data->>'ts')::float8, data->'amplitudes', data->'phases'"
            " from tcpdump where {} and data ? 'ts' order by time"
        ).format(where)
        return self._iter(query, params)

    def get_labels(self, start=None, end=None) -> list[tuple[float, int]]:
        """
        (epoch 초, 라벨) 목록. ts가 없는 라벨(프레임이 이미 지워진 예전 라벨)만
        `time`을 DB_TIMEZONE 기준 시각으로 보고 epoch로 바꾼다.
        """
        where, params = time_filter(start, end)
        query = sql.SQL(
            "select coalesce(ts, extract(epoch from time at time zone %(tz)s)::float8), label"
            " from label where {} order by time"
        ).format(where)
        with self.conn.cursor() as cursor:
            cursor.execute(query, {**params, "tz": DB_TIMEZONE})
            return cursor.fetchall()

    def get_frame_times(self, start=None, end=None) -> list:
        """라벨링할 opencv 프레임의 time 목록 (프레임 이미지는 get_frame으로 하나씩 읽는다)."""
        where, params = time_filter(start, end)
        query = sql.SQL("select time from opencv where {} order by time").format(where)
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return [row[0] for row in cursor.fetchall()]

    def get_frame(self, time) -> dict | None:
        with self.conn.cursor() as cursor:
            cursor.execute("select data from opencv where time = %s limit 1", (time,))
            row = cursor.fetchone()
        return row[0] if row else None

    def insert_labels(self, frames: list[tuple], value: int):
        """frames: (opencv time, 프레임 ts) 목록에 같은 라벨을 붙인다."""
        with self.conn.cursor() as cursor:
            cursor.executemany(
                "insert into label (time, ts, label) values (%s, %s, %s) "
                "on conflict (time) do update set label = excluded.label, ts = excluded.ts",
                [(time, ts, value) for time, ts in frames],
            )
        self.conn.commit()

    def delete_old(self, table: str, days: float, keep_labeled_seconds: float | None) -> int:
        """
        `time`이 days일보다 오래된 행을 지운다.

        :param keep_labeled_seconds: None이 아니면 라벨 앞뒤 이 초 이내의 행은 남긴다
            (학습에 쓰는 신호 데이터 보존)
        :return: 지운 행 수
        """
        query = sql.SQL("delete from {} t where t.time < localtimestamp - %(age)s").format(
            sql.Identifier(table)
        )
        params = {"age": datetime.timedelta(days=days)}
        if keep_labeled_seconds is not None:
            query += sql.SQL(
                " and not exists (select 1 from label l"
                " where l.time between t.time - %(margin)s and t.time + %(margin)s)"
            )
            params["margin"] = datetime.timedelta(seconds=keep_labeled_seconds)
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            deleted = cursor.rowcount
        self.conn.commit()
        return deleted
