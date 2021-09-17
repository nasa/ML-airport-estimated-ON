-- New implementation from Dan Weseley
select distinct on (gufi)
	f.gufi, 
	f."timestamp",
	f.arrival_runway_actual_time, 
	p.position_timestamp, 
	p.position_altitude, 
	p.position_latitude,
	p.position_longitude,
	p.last_update_source as position_source,
	p.position_speed,
	p.position_heading
from matm_flight f, matm_position_all p
where f."timestamp" between :start_time and :end_time
	and p."timestamp" between :start_time and :end_time
	and f.gufi = p.gufi
	and f.arrival_aerodrome_icao_name = :arrival_airport_icao
	and f.arrival_runway_actual_time is not null
	and f.last_update_source = 'SMES'
	and abs(extract(epoch from f.arrival_runway_actual_time - p.position_timestamp)) < 10
	and p.position_latitude is not null
	and p.position_longitude is not null
order by gufi, abs(extract(epoch from f.arrival_runway_actual_time - p.position_timestamp)), p.last_update_source;