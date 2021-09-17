select
	tea.gufi,
	tea."timestamp",
	tea.arrival_meter_fix,
	-- tea.arrival_meter_fix_eta,
	tea.arrival_runway,
	-- tea.arrival_runway_assignment_frozen,
	tea.arrival_runway_sta,
	tea.arrival_runway_eta,
	-- tea.arrival_runway_tracon_assigned,
	-- tea.arrival_scheduling_fix,
	-- tea.arrival_scheduling_fix_eta,
	-- tea.arrival_scheduling_fix_sta,
	-- tea.arrival_scheduling_suspended,
	-- tea.arrival_stas_frozen,
	-- tea.tma_id,
	-- tea.canceled_swim_release_time,
	-- tea.etm,
	-- tea.std,
	-- tea.system_id,
	-- tea.arrival_tracon,
	-- tea.arrival_gate,
	-- tea.arrival_configuration,
	tea.arrival_stream_class
	-- tea.ctm
from tbfm_extension_all tea
left join matm_flight_summary mfs on tea.gufi = mfs.gufi
where
	tea."timestamp" between :start_time and :end_time
	and mfs.arrival_aerodrome_icao_name = :arrival_airport_icao
	and tea.sync_message = false
