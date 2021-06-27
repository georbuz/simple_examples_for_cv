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

/* CTE с лучшим фильмом */
WITH best_film AS (
  SELECT
    genre
  , primaryTitle
  , averageRating
  FROM
    title_basics
    JOIN title_ratings
      USING (tconst)
  WHERE 1=1
    AND genre = 'Comedy'	-- тут выбираем фильм
  /* такая конструкция с order by -> limit 1, отработала быстрее, чем варианты с оконными функциями */
  ORDER BY
    averageRating DESC
  , numVotes DESC
  , tconst ASC
  LIMIT 1
)
SELECT
  genre                                                  AS genre
, COUNT(tb.tconst)                                       AS num_films
, SUM(tr.averageRating * tr.numVotes) / sum(tr.numVotes) AS avg_weighted_rating
, best_film.primaryTitle                                 AS best_film_title
, best_film.averageRating                                AS best_film_rating
FROM
  title_basics tb
  LEFT JOIN title_ratings tr	-- LEFT т.к. не для всех фильмов есть рейтинги
    USING (tconst)
  JOIN best_film
    USING (genre)
GROUP BY
  tb.genre
, best_film.primaryTitle
, best_film.averageRating
;