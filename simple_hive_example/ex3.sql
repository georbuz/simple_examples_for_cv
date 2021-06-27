/* Скрипт инициализация */
CREATE TABLE title_basics
USING PARQUET 
PARTITIONED BY (genre)
CLUSTERED BY (tconst)
INTO 4 BUCKETS
AS 
SELECT 
  tconst
, primaryTitle
, explode(split(genres, '[,]', -1)) AS genre 
FROM title_basics_csv
;

ANALYZE TABLE title_basics COMPUTE STATISTICS;

CREATE TABLE title_crew 
USING PARQUET
CLUSTERED BY (tconst) INTO 4 BUCKETS AS
WITH explode_directors AS (
  SELECT
	tconst
  , explode(split(directors, '[,]', -1)) AS director
  , writers
  FROM title_crew_csv
)
SELECT
  tconst
, director
, explode(split(writers, '[,]', -1))   AS writer
FROM explode_directors
;

ANALYZE TABLE title_crew COMPUTE STATISTICS;


CREATE TABLE name_basics 
USING PARQUET
CLUSTERED BY (nconst) 
INTO 4 BUCKETS 
AS
SELECT * FROM name_basics_csv
;

ANALYZE TABLE name_basics COMPUTE STATISTICS;

CREATE TABLE title_akas 
USING PARQUET
CLUSTERED BY (titleId) 
INTO 4 BUCKETS 
AS
SELECT * FROM title_akas_csv
;

ANALYZE TABLE title_akas COMPUTE STATISTICS;


/* Скрипт запрос */
SELECT
  genre
, primaryName
, primaryTitle
, title
FROM 
  title_basics
  JOIN title_crew 
    USING (tconst)
    JOIN name_basics 
      ON title_crew.director = name_basics.nconst
  LEFT JOIN title_akas 
    ON title_basics.tconst = title_akas.TitleId 
WHERE 1=1
  /* два фильтра ниже по логике скорее условие join-а
   * но для удобочитаемости я перенес их в блок WHERE.
   * а говоря об оптимизации, насколько я знаю,
   * оптимизатор все равно применит эти фильтры на этапе join-а
   * так что лишние данные не будут тащиться по запросу
  */
  AND title_crew.director = title_crew.writer
  AND title_akas.region   = 'RU'
  AND genre = 'Comedy'
 ;