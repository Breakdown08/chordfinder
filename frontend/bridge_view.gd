class_name BridgeView
extends Control
## Draws the bridge game and owns the start button. No rules here: BridgeGame decides everything.

const GAP_STEP: float = 220.0  # distance between gaps, px
const GAP_WIDTH: float = 90.0
const WALKER_X: float = 240.0  # screen x of the walker; the world scrolls under it
const DECK_Y: float = 330.0
const DECK_HEIGHT: float = 22.0
const BOARD_HEIGHT: float = 12.0
const VISIBLE_AHEAD: int = 3
const SIGN_SIZE: int = 56
const TEXT_SIZE: int = 24
const BIG_SIZE: int = 64
const BIG_Y: float = 150.0  # above the row of signs
const BUTTON_DROP: float = 120.0  # below the deck: the centre of the screen belongs to the signs
const BAR_STRIP: float = 110.0  # the instrument's strip at the bottom, kept clear
const SWAY: float = 4.0
const WALK_RATE: float = 14.0  # sway phase per span

const SKY: Color = Color(0.13, 0.17, 0.27)
const ABYSS: Color = Color(0.05, 0.05, 0.09)
const SHORE: Color = Color(0.32, 0.27, 0.2)
const BOARD: Color = Color(0.7, 0.5, 0.28)
const GROWING: Color = Color(0.95, 0.72, 0.25)
const CURRENT: Color = Color(1, 1, 1, 0.14)
const WALKER: Color = Color(0.92, 0.92, 0.95)
const TEXT: Color = Color(0.9, 0.9, 0.92)
const GOOD: Color = Color(0.35, 0.8, 0.5)
const BAD: Color = Color(0.9, 0.4, 0.35)

var game: BridgeGame = BridgeGame.new()
var heard: String = ""  # reply.chord of the last reply, "" = silence
var start_button: Button = Button.new()  # created and added in _init()
var _listening: bool = false


func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	start_button.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	start_button.grow_horizontal = Control.GROW_DIRECTION_BOTH
	start_button.grow_vertical = Control.GROW_DIRECTION_BOTH
	start_button.custom_minimum_size = Vector2(240, 64)
	start_button.offset_top += BUTTON_DROP
	start_button.offset_bottom += BUTTON_DROP
	start_button.visible = false
	start_button.text = "Начать"
	start_button.pressed.connect(begin)
	add_child(start_button)
	visibility_changed.connect(_focus)


func set_listening(on: bool) -> void:
	_listening = on
	_refresh()


func begin() -> void:
	game = BridgeGame.new()
	game.start()
	heard = ""
	print("game: start")  # e2e tests and tuning read the console
	_refresh()


func on_reply(reply: Reply) -> void:
	heard = reply.chord
	var target: String = game.target()
	if game.hear(reply):
		print("game: laid %d %s" % [game.laid, target])
	queue_redraw()


func tick(delta: float) -> void:
	var before: int = game.attempts
	var was: BridgeGame.State = game.state
	var target: String = game.target()
	game.advance(delta)
	if game.attempts < before:
		print("game: fell %d %s left %d" % [game.laid, target, game.attempts])
	if was != BridgeGame.State.WON and game.state == BridgeGame.State.WON:
		print("game: won")
	_refresh()


func attempts_label() -> String:
	return "Попытки: %d" % game.attempts


func _refresh() -> void:
	var idle: bool = game.state in [BridgeGame.State.READY, BridgeGame.State.WON, BridgeGame.State.FALLEN]
	start_button.visible = _listening and idle
	start_button.text = "Начать" if game.state == BridgeGame.State.READY else "Ещё раз"
	_focus()
	queue_redraw()


func _focus() -> void:
	if start_button.visible and is_inside_tree():
		start_button.grab_focus()  # hands are on the guitar: Space presses it


func _gap_x(index: int) -> float:
	return WALKER_X + (float(index + 1) - game.position) * GAP_STEP


func _draw() -> void:
	var font: Font = ThemeDB.fallback_font
	var field: float = maxf(size.y - BAR_STRIP, 0.0)
	var sky_h: float = minf(DECK_Y, field)
	draw_rect(Rect2(0, 0, size.x, sky_h), SKY)
	draw_rect(Rect2(0, sky_h, size.x, maxf(field - sky_h, 0.0)), ABYSS)

	# Shores as strips along the deck, a gap breaks them
	var n: int = game.gaps.size()
	for i: int in n + 1:
		var seg_start: float = _gap_x(i - 1) + GAP_WIDTH * 0.5 if i > 0 else -GAP_STEP
		var seg_end: float = _gap_x(i) - GAP_WIDTH * 0.5 if i < n else size.x + GAP_STEP
		if seg_end > 0.0 and seg_start < size.x:
			draw_rect(Rect2(seg_start, DECK_Y, seg_end - seg_start, DECK_HEIGHT), SHORE)

	for i: int in n:
		var gx: float = _gap_x(i)
		if gx < -GAP_WIDTH or gx > size.x + GAP_WIDTH:
			continue
		var left: float = gx - GAP_WIDTH * 0.5
		draw_rect(Rect2(left, DECK_Y, GAP_WIDTH, DECK_HEIGHT), ABYSS)  # the break
		if i < game.laid:
			draw_rect(Rect2(left, DECK_Y, GAP_WIDTH, BOARD_HEIGHT), BOARD)
		elif i == game.laid:
			draw_rect(Rect2(left - 6.0, DECK_Y - 70.0, GAP_WIDTH + 12.0, DECK_HEIGHT + 90.0), CURRENT)
			var grown: float = GAP_WIDTH * game.judge.progress()
			draw_rect(Rect2(left, DECK_Y, grown, BOARD_HEIGHT), GROWING)
		if i - game.laid < VISIBLE_AHEAD + 1 and i >= game.laid:
			var color: Color = TEXT if i == game.laid else Color(TEXT, 0.6)
			draw_string(font, Vector2(left - 40.0, DECK_Y - 90.0), game.gaps[i], HORIZONTAL_ALIGNMENT_CENTER, GAP_WIDTH + 80.0, SIGN_SIZE, color)

	_draw_walker()

	var lines: PackedStringArray = PackedStringArray([
		"%d/%d" % [game.laid, n],
		attempts_label(),
		"%.1f с на пролёт" % BridgeGame.seconds_per_span(int(game.position)),
		"слышу: %s" % (heard if heard != "" else "…"),
	])
	for j: int in lines.size():
		draw_string(font, Vector2(16.0, 80.0 + 32.0 * j), lines[j], HORIZONTAL_ALIGNMENT_LEFT, -1, TEXT_SIZE, TEXT)
	var fell: bool = game.state in [BridgeGame.State.RETURNING, BridgeGame.State.FALLEN]
	if game.state == BridgeGame.State.WON:
		draw_string(font, Vector2(0.0, BIG_Y), "Мост пройден", HORIZONTAL_ALIGNMENT_CENTER, size.x, BIG_SIZE, GOOD)
	elif fell:
		draw_string(font, Vector2(0.0, BIG_Y), "Провал: %s" % game.target(), HORIZONTAL_ALIGNMENT_CENTER, size.x, BIG_SIZE, BAD)


func _draw_walker() -> void:
	var x: float = WALKER_X
	var y: float = DECK_Y
	var phase: float = game.position * WALK_RATE
	var moving: bool = game.state == BridgeGame.State.RUNNING
	if game.state == BridgeGame.State.FALLEN:
		x = _gap_x(game.laid)
		y = DECK_Y + 90.0
	elif game.state == BridgeGame.State.RETURNING:
		var share: float = 1.0 - game.return_left / BridgeGame.RETURN_SECONDS
		x = _gap_x(game.laid)
		y = DECK_Y + 90.0 * share
	elif moving:
		x += sin(phase) * SWAY
	var tilt: float = sin(phase) * 3.0 if moving else 0.0
	draw_rect(Rect2(x - 8.0 + tilt, y - 56.0, 16.0, 36.0), WALKER)  # torso
	draw_circle(Vector2(x + tilt, y - 66.0), 10.0, WALKER)  # head
	draw_line(Vector2(x, y - 20.0), Vector2(x - 8.0 + sin(phase) * 6.0, y), WALKER, 4.0)
	draw_line(Vector2(x, y - 20.0), Vector2(x + 8.0 - sin(phase) * 6.0, y), WALKER, 4.0)
