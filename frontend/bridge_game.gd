class_name BridgeGame
extends RefCounted
## One run of the bridge: the walker crosses gaps, a right chord lays a board. Pure state, time comes as an argument.

enum State { READY, RUNNING, RETURNING, WON, FALLEN }

# Flat on purpose: in 4.5.1 a const Array[PackedStringArray] compiles but ends up empty
const SEQUENCE: PackedStringArray = [
	"C", "G", "Am", "F",  # I-V-vi-IV, C major
	"Am", "Dm", "G", "C",  # vi-ii-V-I
	"Dm", "Bb", "F", "C",  # i-VI-III-VII, D minor
	"F", "Bdim", "Em", "Am",  # IV-vii°-iii-vi, a stretch of the diatonic circle
]
const SPANS_PER_STEP: int = 4
const START_SECONDS: float = 4.0
const STEP_SECONDS: float = 0.5
const MIN_SECONDS: float = 2.5
const MAX_DELTA: float = 0.1  # a hidden tab must not kill the walker in one frame
const ATTEMPTS: int = 3
const RETURN_SECONDS: float = 1.5  # longer than the model window (1 s): else sound from before the fall stays in it

var state: State = State.READY
var position: float = 0.0  # in spans; gap i stands at float(i + 1)
var laid: int = 0  # boards laid, also the index of the current gap
var attempts: int = ATTEMPTS  # left, the current one included; 0 only when FALLEN
var return_left: float = 0.0  # seconds until the return ends; not 0 only when RETURNING
var gaps: PackedStringArray
var judge: ChordJudge = ChordJudge.new()


func _init(chords: PackedStringArray = SEQUENCE) -> void:
	gaps = chords


static func seconds_per_span(span: int) -> float:
	return maxf(START_SECONDS - STEP_SECONDS * (span / SPANS_PER_STEP), MIN_SECONDS)


static func wanted(search: String) -> bool:
	return "game" in search and not "measure" in search


func start() -> void:
	if state == State.READY:
		state = State.RUNNING


func target() -> String:
	return gaps[laid] if laid < gaps.size() else ""


func listening() -> bool:
	return state == State.RUNNING and laid < gaps.size() and position >= float(laid)


## True if a board was laid.
func hear(reply: Reply) -> bool:
	if not listening():
		return false
	if judge.hear(reply, target()):
		laid += 1
		judge.reset()
		return true
	return false


func advance(delta: float) -> void:
	var step: float = minf(delta, MAX_DELTA)
	if state == State.RETURNING:
		return_left -= step
		if return_left <= 0.0:
			return_left = 0.0
			state = State.RUNNING
		return
	if state != State.RUNNING:
		return
	position += step / seconds_per_span(int(position))
	if laid < gaps.size() and position >= float(laid + 1):
		attempts -= 1
		judge.reset()
		if attempts > 0:
			state = State.RETURNING
			position = float(laid)
			return_left = RETURN_SECONDS
		else:
			state = State.FALLEN
			position = float(laid + 1)
	elif position >= float(gaps.size() + 1):
		state = State.WON
		position = float(gaps.size() + 1)
