-- PostgreSQL 컨테이너 최초 기동 시 실행된다 (docker-entrypoint-initdb.d).
-- fluent-bit pgsql 출력은 (tag, time, data) 형식으로 기록하며, time은 DB_TIMEZONE 기준 시각이다.
CREATE TABLE IF NOT EXISTS audio (tag varchar, time timestamp, data jsonb);
CREATE TABLE IF NOT EXISTS tcpdump (tag varchar, time timestamp, data jsonb);
CREATE TABLE IF NOT EXISTS opencv (tag varchar, time timestamp, data jsonb);
CREATE INDEX IF NOT EXISTS audio_time_idx ON audio (time);
CREATE INDEX IF NOT EXISTS tcpdump_time_idx ON tcpdump (time);
CREATE INDEX IF NOT EXISTS opencv_time_idx ON opencv (time);

-- labeler.py가 opencv 프레임의 time 값을 그대로 키로 사용한다 (0: 평상시, 1: 낙상)
CREATE TABLE IF NOT EXISTS label (time timestamp PRIMARY KEY, label smallint NOT NULL);
