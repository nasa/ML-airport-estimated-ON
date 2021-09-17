select
    gufi,
    "timestamp",
    route_text,
    fix_list_values 
from matm_flight
where 
    "timestamp" between :start_time and :end_time
    and arrival_aerodrome_icao_name = :arrival_airport_icao
    and route_text is not null
order by gufi, "timestamp"