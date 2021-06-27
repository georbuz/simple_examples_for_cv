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
SELECT
  genre       	AS genre
, COUNT(tconst)	AS num_films
, MAX(numVotes) AS max_num_votes
FROM 
  title_basics
  JOIN title_ratings	-- inner join, т.к. нам обязательно нужно поле из таблицы рейтингов
    USING (tconst)
WHERE 1=1
  AND genre 		 = 'Comedy' -- выбираем жанр
  AND averageRating >= 4		-- выбираем нижний рейтинг
GROUP BY
  genre
 ;