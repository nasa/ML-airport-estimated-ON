with mfs as (
    select 
        gufi,
        aircraft_engine_class,
        aircraft_type,
        major_carrier,
        carrier,
        flight_type,
        international,
        arrival_aerodrome_icao_name,
        COALESCE(arrival_stand_actual,
            arrival_stand_user,
            arrival_stand_airline) as arrival_stand_actual,
        COALESCE(arrival_stand_actual_time,
            arrival_stand_airline_time) as arrival_stand_actual_time,
        arrival_stand_initial_time,
        arrival_movement_area_actual_time,
        COALESCE(arrival_runway_actual,
            arrival_runway_user,
            arrival_runway_assigned,
            arrival_runway_airline) as arrival_runway_actual,
        arrival_runway_actual_time,
        arrival_runway_scheduled_time,
        COALESCE(departure_stand_actual_time,
            departure_stand_airline_time) as departure_stand_actual_time,
        departure_stand_initial_time,
        departure_movement_area_actual_time,
        COALESCE(departure_runway_actual,
            departure_runway_user,
            departure_runway_assigned,
            departure_runway_airline) as departure_runway_actual,
        departure_runway_actual_time,
        departure_runway_scheduled_time,
        estimated_departure_clearance_time,
        departure_runway_metered_time_value
    from 
        matm_flight_summary
    where 
        (
            arrival_aerodrome_icao_name = :arrival_airport_icao 
            and arrival_runway_actual_time between :start_time and :end_time
        )
        and cancelled is null
        and sensitive_data is not TRUE
)
select
    mfs.*,
    extract(epoch from mfs.departure_runway_actual_time - mfs.departure_runway_scheduled_time) as actual_departure_runway_delay_vs_schedule,
    extract(epoch from mfs.departure_runway_actual_time - mfs.estimated_departure_clearance_time) as actual_departure_runway_EDCT_compliance,
    extract(epoch from mfs.departure_runway_actual_time - mfs.departure_runway_metered_time_value) as actual_departure_runway_APREQ_compliance
from 
    mfs
order by arrival_runway_actual_time