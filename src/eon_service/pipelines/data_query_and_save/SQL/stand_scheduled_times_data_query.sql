with a as (
select distinct on (gufi)
	gufi,
	"timestamp" as arrival_stand_scheduled_time_timestamp,
	arrival_stand_airline_time as arrival_stand_scheduled_time
from matm_flight
where "timestamp" between :start_time and :end_time
	and arrival_aerodrome_icao_name = :arrival_airport_icao
	and arrival_stand_airline_time is not null
	and arrival_stand_airline_time > "timestamp"
order by gufi, "timestamp"
),
d as (
select distinct on (gufi)
	gufi,
	"timestamp" as departure_stand_scheduled_time_timestamp,
	departure_stand_airline_time as departure_stand_scheduled_time
from matm_flight
where "timestamp" between :start_time and :end_time
	and arrival_aerodrome_icao_name = :arrival_airport_icao
	and departure_stand_airline_time is not null
	and departure_stand_airline_time > "timestamp"
order by gufi, "timestamp"
)
select 
    coalesce(a.gufi, d.gufi) as gufi,
    d.departure_stand_scheduled_time_timestamp,
    d.departure_stand_scheduled_time,
    a.arrival_stand_scheduled_time_timestamp,
    a.arrival_stand_scheduled_time
from a
full outer join d
on a.gufi=d.gufi