select
    airport_id,
    start_time as "timestamp",
    arrival_runways,
    departure_runways,
    weather_report
from datis_parser_message
where 
    airport_id = :arrival_airport_icao
    and datis_time between (timestamp :start_time - interval '24 hours') and (timestamp :end_time + interval '24 hours')
    and start_time between :start_time and :end_time
order by start_time
