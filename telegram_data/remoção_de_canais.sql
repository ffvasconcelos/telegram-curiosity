SELECT t1.*
FROM messages t1
INNER JOIN (
  SELECT channel_id
  FROM messages
  GROUP BY channel_id
  HAVING COUNT(DISTINCT from_id) > 1
) t2 ON t1.channel_id = t2.channel_id
