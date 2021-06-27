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

/* Скрипт запрос */
SELECT
  genre       	AS genre
, COUNT(tconst)	AS num_films
FROM title_basics
WHERE 1=1
  AND genre = 'Comedy' -- выбираем жанр
GROUP BY
  genre
 ;