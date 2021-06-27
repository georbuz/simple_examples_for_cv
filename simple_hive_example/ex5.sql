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


CREATE TABLE name_basics 
USING PARQUET
CLUSTERED BY (nconst) 
INTO 4 BUCKETS 
AS
SELECT * FROM name_basics_csv
;

ANALYZE TABLE name_basics COMPUTE STATISTICS;


CREATE TABLE title_crew 
USING PARQUET
CLUSTERED BY (tconst) 
INTO 4 BUCKETS 
AS
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


CREATE TABLE title_principals
USING PARQUET
PARTITIONED BY (category)
CLUSTERED BY (tconst)
INTO 4 BUCKETS
AS
SELECT 
  tconst
, nconst
, CAST(ordering AS INT) AS ordering
, category
, job
, characters 
FROM title_principals_csv
;

ANALYZE TABLE title_principals COMPUTE STATISTICS;

CREATE TABLE title_ratings 
USING PARQUET 
CLUSTERED BY (tconst)	-- кластеризовать имеет смысл по ключам соединения
INTO 4 BUCKETS
AS 
SELECT 
  tconst								AS tconst
, CAST(averageRating AS DECIMAL(2, 1))	AS averageRating -- рейтинг это число от 0 до 10 с одной цифрой после запятой
, CAST(numVotes AS INT) 				AS numVotes		 -- макс. значение по полю numVotes ~ 100 000, так что кастуем в инт4
FROM title_ratings_csv
;

ANALYZE TABLE title_ratings COMPUTE STATISTICS;


/* Скрипт запрос */

/* В CTE считаем агрегаты по всем персонам, которые могут понадобиться в теле запроса */
WITH aggregates AS (
  SELECT
    nconst                                        AS nconst
  , COUNT(title_basics.tconst)                    AS num_films
  , MAX(numVotes)                                 AS max_votes
  , SUM(averageRating * numVotes) / sum(numVotes) AS avg_weighted_rating
  FROM
  title_basics
    JOIN title_principals
      USING (tconst)
    LEFT JOIN title_ratings
      USING (tconst)
  WHERE category IN ('director', 'actress', 'actor', 'self') -- берем только нужные категории
  GROUP BY
    nconst
)
SELECT
  nb1.primaryName                               AS director_name
, dir_info.num_films                            AS director_num_films
, dir_info.avg_weighted_rating                  AS director_avg_rating
, dir_info.max_votes                            AS director_max_votes
, nb2.primaryName                               AS fav_name
, act_info.num_films                            AS fav_num_films
, act_info.avg_weighted_rating                  AS fav_avg_rating
, act_info.max_votes                            AS fav_max_votes
, COUNT(title_basics.tconst)                    AS num_films
, MAX(numVotes)                                 AS max_votes
, SUM(averageRating * numVotes) / sum(numVotes) AS avg_weighted_rating
/* сначала собирается инфа для пары (режиссер, любимчик) */
FROM
title_basics
  LEFT JOIN title_ratings
    USING (tconst)
  JOIN title_crew
    USING (tconst)
    JOIN name_basics nb1
      ON title_crew.director = nb1.nconst
  JOIN title_principals
    USING (tconst)
    JOIN name_basics nb2
      ON title_principals.nconst = nb2.nconst
/* здесь цепляются предварительно рассчитанные агрегаты */
  JOIN aggregates dir_info	-- для режиссера
    ON nb1.nconst = dir_info.nconst
  JOIN aggregates act_info	-- для актера
    ON nb2.nconst = act_info.nconst
WHERE 1=1
  AND title_crew.director = 'nm0617588'							-- тут подставляем id режиссера
  AND title_principals.category IN ('actress', 'actor', 'self')	-- берем только нужные категории
GROUP BY
  nb1.primaryName
, nb2.primaryName
, dir_info.num_films
, dir_info.avg_weighted_rating
, dir_info.max_votes
, act_info.num_films
, act_info.avg_weighted_rating
, act_info.max_votes