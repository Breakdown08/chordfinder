class_name Bars
extends Control
## The instrument: what the model hears in the current window, before the vote.
## One column per chord (raw probability), the voting threshold across them, input level on the right.

const MIN_CONFIDENCE: float = 0.6  # must match backend stream.MIN_CONFIDENCE
const SILENCE_RMS: float = 0.01  # must match backend audio.SILENCE_RMS
const FLOOR_DB: float = -60.0
const METER_WIDTH: float = 20.0
const GAP: float = 12.0
const LABEL_HEIGHT: float = 28.0
const FONT_SIZE: int = 20

const IDLE: Color = Color(0.45, 0.47, 0.52)
const TOP: Color = Color(0.95, 0.72, 0.25)  # the window's best guess, not yet a shown chord
const SHOWN: Color = Color(0.35, 0.8, 0.5)  # the chord the vote let through
const GUIDE: Color = Color(1, 1, 1, 0.35)
const TEXT: Color = Color(0.9, 0.9, 0.92)
const TRACK: Color = Color(1, 1, 1, 0.06)

var _names: PackedStringArray = PackedStringArray()  # kept through silence so the columns stay in place
var _reply: Reply = Reply.new()
var _shown: String = ""


## Position of an RMS on the level meter, 0..1 on a dB scale: speech-quiet rooms and strums both fit.
static func level(rms: float) -> float:
	if rms <= 0.0:
		return 0.0
	return clampf(1.0 - linear_to_db(rms) / FLOOR_DB, 0.0, 1.0)


func show_reply(reply: Reply, shown: String) -> void:
	if not reply.probabilities.is_empty():
		_names = PackedStringArray(reply.probabilities.keys())
	_reply = reply
	_shown = shown
	queue_redraw()


func _draw() -> void:
	var font: Font = ThemeDB.fallback_font
	var plot: float = size.y - LABEL_HEIGHT
	var meter_x: float = size.x - METER_WIDTH
	var n: int = _names.size()
	var column: float = (meter_x - GAP * (n + 1)) / maxf(n, 1)

	for i: int in n:
		var chord: String = _names[i]
		var x: float = GAP + i * (column + GAP)
		var p: float = _reply.probabilities.get(chord, 0.0)
		var color: Color = SHOWN if chord == _shown else TOP if chord == _reply.chord else IDLE
		draw_rect(Rect2(x, 0, column, plot), TRACK)
		draw_rect(Rect2(x, plot * (1.0 - p), column, plot * p), color)
		draw_string(font, Vector2(x, size.y - 4), chord, HORIZONTAL_ALIGNMENT_CENTER, column, FONT_SIZE, TEXT)

	var threshold: float = plot * (1.0 - MIN_CONFIDENCE)
	draw_dashed_line(Vector2(0, threshold), Vector2(meter_x - GAP, threshold), GUIDE, 2.0, 8.0)

	var height: float = plot * Bars.level(_reply.rms)
	var heard: bool = _reply.rms >= SILENCE_RMS
	draw_rect(Rect2(meter_x, 0, METER_WIDTH, plot), TRACK)
	draw_rect(Rect2(meter_x, plot - height, METER_WIDTH, height), SHOWN if heard else IDLE)
	var floor_y: float = plot * (1.0 - Bars.level(SILENCE_RMS))
	draw_line(Vector2(meter_x - 4, floor_y), Vector2(size.x + 4, floor_y), GUIDE, 2.0)
