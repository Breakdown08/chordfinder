class_name Measure
extends RefCounted
## A measuring session: each chord in turn, a button starts its take, the server counts the strums.
## Open the page with ?measure; the server needs CHORDFINDER_SESSIONS (backend/src/chordfinder/measure.py).

enum State { WAITING, RECORDING, DONE }

const CHORDS: PackedStringArray = ["Am", "Bb", "Bdim", "C", "Dm", "Em", "F", "G"]  # model classes
const STRUMS: int = 3  # must match backend tools/takes.py
const TAIL_SECONDS: float = 2.0  # the last strum has to ring out on the record

var state: State = State.WAITING
var index: int = 0
var strums: int = 0
var _last_strum_at: float = -1.0


func chord() -> String:
	return CHORDS[mini(index, CHORDS.size() - 1)]


func record() -> void:
	state = State.RECORDING
	_reset()


func heard(count: int, now: float) -> void:
	if state != State.RECORDING:
		return
	strums = count
	if strums >= STRUMS and _last_strum_at < 0.0:
		_last_strum_at = now


func tail_over(now: float) -> bool:
	return state == State.RECORDING and _last_strum_at >= 0.0 and now - _last_strum_at >= TAIL_SECONDS


func advance() -> void:
	index += 1
	state = State.DONE if index >= CHORDS.size() else State.WAITING
	_reset()


## During a take: start this chord over. Between takes: go back to the previous one.
func redo() -> void:
	if state != State.RECORDING:
		index = maxi(index - 1, 0)
	state = State.WAITING
	_reset()


func text() -> String:
	var progress: String = "%d/%d" % [mini(index + 1, CHORDS.size()), CHORDS.size()]
	match state:
		State.WAITING:
			return "%s %s: пробел" % [progress, chord()]
		State.RECORDING:
			return "%s %s: удар %d из %d" % [progress, chord(), mini(strums, STRUMS), STRUMS]
	return "Готово"


## The big counter, readable from behind a guitar.
func counter() -> String:
	return "%s  %d/%d" % [chord(), mini(strums, STRUMS), STRUMS] if state == State.RECORDING else ""


func _reset() -> void:
	strums = 0
	_last_strum_at = -1.0
