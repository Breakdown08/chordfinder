extends RefCounted
## Checks 1-55 of the bridge-game spec. Run by test_game.gd; names the game classes directly.

const STEP: float = 0.05
const RESTART: String = "Ещё раз"
const BEGIN: String = "Начать"

var finished: bool = false
var failed: int = 0


func check(ok: bool, what: String) -> void:
	print(("ok   " if ok else "FAIL ") + what)
	if not ok:
		failed += 1


func reply(chord: String, confidence: float = 0.9, rms: float = 0.05) -> Reply:
	var r: Reply = Reply.new()
	r.chord = chord
	r.confidence = confidence
	r.rms = rms
	return r


func silence() -> Reply:
	return Reply.new()


## n replies of one chord; true if any of them returned true.
func any_true(game: BridgeGame, chord: String, n: int) -> bool:
	var got: bool = false
	for i: int in n:
		if game.hear(reply(chord)):
			got = true
	return got


## A full series; returns the result of the last hear.
func series(game: BridgeGame, chord: String) -> bool:
	var last: bool = false
	for i: int in ChordJudge.CONFIRM_REPLIES:
		last = game.hear(reply(chord))
	return last


func judge_series(judge: ChordJudge, chord: String, n: int) -> bool:
	var got: bool = false
	for i: int in n:
		if judge.hear(reply(chord), "C"):
			got = true
	return got


func fall(game: BridgeGame) -> float:
	var total: float = 0.0
	for i: int in 1000:
		if game.state != BridgeGame.State.RUNNING:
			break
		game.advance(STEP)
		total += STEP
	return total


func recover(game: BridgeGame) -> float:
	var total: float = 0.0
	for i: int in 1000:
		if game.state != BridgeGame.State.RETURNING:
			break
		game.advance(STEP)
		total += STEP
	return total


func until_listening(game: BridgeGame) -> bool:
	for i: int in 1000:
		if game.listening():
			break
		game.advance(STEP)
	return game.listening()


func until_position(game: BridgeGame, position: float) -> void:
	for i: int in 2000:
		if game.position >= position:
			break
		game.advance(STEP)


func two() -> BridgeGame:
	var game: BridgeGame = BridgeGame.new(PackedStringArray(["C", "G"]))
	game.start()
	return game


func span_sum() -> float:
	var total: float = 0.0
	for i: int in 17:
		total += BridgeGame.seconds_per_span(i)
	return total


func view_series(view: BridgeView, chord: String) -> void:
	for i: int in ChordJudge.CONFIRM_REPLIES:
		view.on_reply(reply(chord))


func run() -> void:
	judge_checks()
	sequence_checks()
	tempo_checks()
	course_checks()
	attempt_checks()
	view_checks()
	compile_checks()
	finished = true


func judge_checks() -> void:
	var n: int = ChordJudge.CONFIRM_REPLIES
	var j: ChordJudge = ChordJudge.new()

	check(n >= 2, "1: one reply never confirms a chord")

	var early: bool = judge_series(j, "C", n - 1)
	check(not early and j.count == n - 1, "2: CONFIRM_REPLIES - 1 replies: false, count n-1")
	check(j.hear(reply("C"), "C"), "2: the next reply confirms")

	j = ChordJudge.new()
	judge_series(j, "C", n - 1)
	j.hear(reply("Em"), "C")
	check(j.count == 0, "3: another chord resets the count")

	j = ChordJudge.new()
	judge_series(j, "C", n - 1)
	j.hear(silence(), "C")
	check(j.count == 0, "4: silence resets the count")

	j = ChordJudge.new()
	j.hear(reply("C", ChordJudge.MIN_CONFIDENCE), "C")
	check(j.count == 1, "5: confidence == MIN_CONFIDENCE counts")
	j.hear(reply("C", ChordJudge.MIN_CONFIDENCE - 0.01), "C")
	check(j.count == 0, "5: confidence below MIN_CONFIDENCE resets")

	j = ChordJudge.new()
	j.hear(reply("C", 0.9, ChordJudge.MIN_RMS), "C")
	check(j.count == 1, "6: rms == MIN_RMS counts")
	j.hear(reply("C", 0.9, ChordJudge.MIN_RMS - 0.001), "C")
	check(j.count == 0, "6: rms below MIN_RMS resets")

	j = ChordJudge.new()
	var ever: bool = false
	for i: int in 20:
		if j.hear(silence(), ""):
			ever = true
		if j.hear(reply("", 0.9, 0.05), ""):
			ever = true
	check(not ever and j.count == 0, "7: silence against an empty target never counts")

	j = ChordJudge.new()
	var r: Reply = reply("Em")
	r.stable = "C"
	var by_stable: bool = false
	for i: int in n * 2:
		if j.hear(r, "C"):
			by_stable = true
	check(not by_stable and j.count == 0, "8: stable == target does not count when chord differs")
	j = ChordJudge.new()
	r = reply("C")
	r.stable = "Em"
	var by_chord: bool = false
	for i: int in n:
		by_chord = j.hear(r, "C")
	check(by_chord, "8: chord == target counts regardless of stable")

	j = ChordJudge.new()
	check(j.progress() == 0.0, "9: progress is 0.0 at the start")
	j.hear(reply("C"), "C")
	check(is_equal_approx(j.progress(), 1.0 / float(n)), "9: progress is count / CONFIRM_REPLIES")
	judge_series(j, "C", n - 1)
	check(j.progress() == 1.0, "9: progress is 1.0 on a full series")
	judge_series(j, "C", 5)
	check(j.progress() <= 1.0, "9: progress never exceeds 1.0")
	j.reset()
	check(j.count == 0 and j.progress() == 0.0, "9: reset() clears the count")


func sequence_checks() -> void:
	var seq: PackedStringArray = BridgeGame.SEQUENCE
	check(seq.size() == 16, "10: SEQUENCE has 16 gaps")
	var known: bool = true
	var distinct: Dictionary[String, bool] = {}
	for c: String in seq:
		if not Measure.CHORDS.has(c):
			known = false
		distinct[c] = true
	check(known, "10: every SEQUENCE chord is a Measure chord")
	check(distinct.size() == Measure.CHORDS.size(), "10: all model chords appear")

	var differ: bool = true
	for i: int in seq.size() - 1:
		if seq[i] == seq[i + 1]:
			differ = false
	check(differ, "11: neighbouring chords differ")

	check(seq.slice(0, 8) == PackedStringArray(["C", "G", "Am", "F", "Am", "Dm", "G", "C"]), "12: first eight chords")

	check(BridgeGame.new().gaps == seq, "13: default game takes SEQUENCE")
	var custom: PackedStringArray = PackedStringArray(["C", "G"])
	check(BridgeGame.new(custom).gaps == custom, "13: custom gaps are kept")


func tempo_checks() -> void:
	var per: int = BridgeGame.SPANS_PER_STEP
	check(BridgeGame.seconds_per_span(0) == BridgeGame.START_SECONDS, "14: span 0 is START_SECONDS")
	check(BridgeGame.seconds_per_span(per - 1) == BridgeGame.START_SECONDS, "14: last span of the first step")
	check(
		is_equal_approx(BridgeGame.seconds_per_span(per), BridgeGame.START_SECONDS - BridgeGame.STEP_SECONDS),
		"14: the next step is faster by STEP_SECONDS"
	)

	var ok: bool = true
	var prev: float = BridgeGame.seconds_per_span(0)
	for i: int in 101:
		var s: float = BridgeGame.seconds_per_span(i)
		if s > prev or s < BridgeGame.MIN_SECONDS:
			ok = false
		prev = s
	check(ok, "15: tempo never slows and stays at or above MIN_SECONDS")
	check(BridgeGame.seconds_per_span(100) == BridgeGame.MIN_SECONDS, "15: span 100 is MIN_SECONDS")

	var total: float = span_sum()
	check(total >= 30.0 and total <= 90.0, "16: the whole run lasts under a minute and a half, over half a minute")


func course_checks() -> void:
	var g: BridgeGame = BridgeGame.new()
	check(
		g.state == BridgeGame.State.READY and g.position == 0.0 and g.laid == 0 and g.target() == "C",
		"17: a new game is ready at the start"
	)
	check(not g.listening(), "17: a new game does not listen")
	g.advance(0.1)
	check(g.position == 0.0 and g.state == BridgeGame.State.READY, "17: advance does nothing before start")
	check(not any_true(g, "C", ChordJudge.CONFIRM_REPLIES * 2) and g.laid == 0, "17: hear is false before start")

	g = BridgeGame.new()
	g.start()
	check(g.state == BridgeGame.State.RUNNING and g.listening(), "18: start() runs and listens")
	series(g, "C")
	for i: int in 3:
		g.advance(STEP)
	var p: float = g.position
	g.start()
	check(
		g.position == p and g.laid == 1 and g.state == BridgeGame.State.RUNNING,
		"18: a second start() changes nothing"
	)

	g = BridgeGame.new()
	g.start()
	g.advance(STEP)
	check(is_equal_approx(g.position, STEP / BridgeGame.seconds_per_span(0)), "19: position grows by delta / span")

	var a: BridgeGame = BridgeGame.new()
	var b: BridgeGame = BridgeGame.new()
	a.start()
	b.start()
	a.advance(100.0)
	b.advance(BridgeGame.MAX_DELTA)
	check(is_equal_approx(a.position, b.position) and a.position > 0.0, "20: a long delta is cut to MAX_DELTA")

	g = BridgeGame.new()
	g.start()
	var last: bool = series(g, "C")
	check(last and g.laid == 1 and g.judge.count == 0 and g.position == 0.0, "21: a series lays a board in place")

	g = two()
	series(g, "C")
	check(not g.listening(), "22: the window is closed after a board")
	check(not any_true(g, "G", ChordJudge.CONFIRM_REPLIES * 2), "22: answers outside the window are ignored")
	check(g.laid == 1 and g.judge.count == 0, "22: nothing counted outside the window")

	g = two()
	series(g, "C")
	until_position(g, 1.0)
	check(g.listening(), "23: the window opens at position 1.0")
	check(series(g, "G") and g.laid == 2, "23: a series in the open window lays the board")

	g = two()
	series(g, "C")
	var early: bool = false
	for i: int in 2000:
		if g.position >= 1.0:
			break
		if g.hear(reply("G")):
			early = true
		g.advance(STEP)
	check(not early and g.laid == 1 and g.judge.count == 0, "24: chord played before the window counts nothing")
	check(not any_true(g, "G", ChordJudge.CONFIRM_REPLIES - 1), "24: after opening a full series is needed")
	check(g.hear(reply("G")) and g.laid == 2, "24: the last reply of the series lays the board")

	g = BridgeGame.new()
	g.start()
	check(not any_true(g, "Em", 50), "25: wrong chord is never a board")
	check(g.state == BridgeGame.State.RUNNING and g.laid == 0, "25: wrong chord costs no attempt")
	check(series(g, "C") and g.laid == 1, "25: the right series still lays the board")

	g = BridgeGame.new()
	g.start()
	var span: float = fall(g)
	check(
		g.state == BridgeGame.State.RETURNING and g.position == 0.0 and g.laid == 0,
		"26: time over returns to the start of the span"
	)
	check(
		g.attempts == BridgeGame.ATTEMPTS - 1 and g.return_left == BridgeGame.RETURN_SECONDS and g.target() == "C",
		"26: one attempt lost, pause set, same target"
	)
	check(absf(span - BridgeGame.seconds_per_span(0)) < 0.1, "26: the span takes seconds_per_span(0)")

	var plain: BridgeGame = BridgeGame.new()
	plain.start()
	var plain_steps: int = 0
	for i: int in 1000:
		if plain.state != BridgeGame.State.RUNNING:
			break
		plain.advance(STEP)
		plain_steps += 1
	var noisy: BridgeGame = BridgeGame.new()
	noisy.start()
	var noisy_steps: int = 0
	for i: int in 1000:
		noisy.hear(reply("Em"))
		noisy.advance(STEP)
		noisy_steps += 1
		if noisy.state != BridgeGame.State.RUNNING:
			break
	check(
		noisy.state == BridgeGame.State.RETURNING and noisy_steps == plain_steps,
		"27: wrong chords do not bring the fall closer"
	)

	g = BridgeGame.new()
	g.start()
	var near: bool = false
	for i: int in 2000:
		if g.position >= 0.98:
			near = true
			break
		g.advance(STEP)
	check(near and g.position < 1.0, "28: the walker stands just before the first gap")
	check(series(g, "C"), "28: the series lays the board at the last step")
	for i: int in 10:
		g.advance(STEP)
	check(
		g.position > 1.0 and g.state == BridgeGame.State.RUNNING and g.attempts == BridgeGame.ATTEMPTS,
		"28: crossing the boundary with a board costs nothing"
	)

	g = BridgeGame.new()
	g.start()
	var steps: int = 0
	for i: int in 2000:
		if g.state != BridgeGame.State.RUNNING:
			break
		if g.listening():
			series(g, g.target())
		g.advance(STEP)
		steps += 1
	check(g.state == BridgeGame.State.WON, "29: the full run is won")
	check(g.laid == 16 and g.position == 17.0 and g.attempts == BridgeGame.ATTEMPTS, "29: all boards, end of the bridge")
	check(g.target() == "" and not g.listening(), "29: nothing left to target")
	check(absf(float(steps) * STEP - span_sum()) < 0.2, "29: duration matches the sum of spans")

	check(BridgeGame.wanted("?game") and BridgeGame.wanted("?game&mono=1"), "30: ?game wants the game")
	check(
		not BridgeGame.wanted("") and not BridgeGame.wanted("?measure") and not BridgeGame.wanted("?measure&game"),
		"30: other addresses do not"
	)
	check(not BridgeGame.wanted("?voice=1"), "30: ?voice=1 does not")


func attempt_checks() -> void:
	var n: int = ChordJudge.CONFIRM_REPLIES
	var per: int = BridgeGame.SPANS_PER_STEP
	var att: int = BridgeGame.ATTEMPTS
	var g: BridgeGame = BridgeGame.new()
	check(att == 3 and BridgeGame.RETURN_SECONDS > 1.0, "31: three attempts, pause longer than the model window")
	check(g.attempts == att and g.return_left == 0.0, "31: a new game has all attempts")
	g.start()
	check(g.attempts == att and g.return_left == 0.0, "31: start() keeps them")

	g = two()
	fall(g)
	check(not g.listening(), "32: the game is deaf while returning")
	check(not any_true(g, "C", n * 2), "32: replies during the pause give false")
	check(g.laid == 0 and g.judge.count == 0, "32: nothing counted during the pause")
	recover(g)
	check(g.judge.count == 0, "32: the counter is empty after the pause")
	check(not any_true(g, "C", n - 1), "32: after the pause n-1 replies are not enough")
	check(g.hear(reply("C")), "32: the next one lays the board")

	g = two()
	any_true(g, "C", n - 1)
	fall(g)
	check(g.judge.count == 0, "33: the fall resets the counter")
	recover(g)
	check(not any_true(g, "C", n - 1) and g.laid == 0, "33: the old series does not survive")

	g = two()
	fall(g)
	var calls: int = 0
	var steady: bool = true
	for i: int in 1000:
		if g.state != BridgeGame.State.RETURNING:
			break
		g.advance(BridgeGame.MAX_DELTA)
		calls += 1
		if g.position != 0.0 or g.laid != 0 or g.attempts != att - 1:
			steady = false
	var paused: float = float(calls) * BridgeGame.MAX_DELTA
	check(
		paused >= BridgeGame.RETURN_SECONDS - 0.001 and paused <= BridgeGame.RETURN_SECONDS + BridgeGame.MAX_DELTA + 0.001,
		"34: the pause lasts RETURN_SECONDS"
	)
	check(steady, "34: the walker stands still during the pause")
	check(
		g.state == BridgeGame.State.RUNNING and g.return_left == 0.0 and g.position == 0.0,
		"34: the pause ends in RUNNING at the same place"
	)
	check(g.listening() and g.target() == "C", "34: the window is open at once, same target")

	g = two()
	fall(g)
	var before: float = g.return_left
	g.advance(100.0)
	check(g.state == BridgeGame.State.RETURNING, "35: a long delta does not end the pause")
	check(is_equal_approx(before - g.return_left, BridgeGame.MAX_DELTA), "35: the pause is cut to MAX_DELTA")

	g = two()
	fall(g)
	g.start()
	check(
		g.state == BridgeGame.State.RETURNING and g.position == 0.0 and g.laid == 0,
		"36: start() does not touch a return"
	)
	check(g.attempts == att - 1 and g.return_left == BridgeGame.RETURN_SECONDS, "36: attempts and pause unchanged")

	g = BridgeGame.new()
	g.start()
	for k: int in per:
		until_listening(g)
		series(g, BridgeGame.SEQUENCE[k])
	until_listening(g)
	check(g.laid == per, "37: boards laid up to the far gap")
	fall(g)
	check(g.position == float(per) and g.laid == per, "37: return to the start of the same span")
	check(g.target() == BridgeGame.SEQUENCE[per], "37: same target")
	recover(g)
	var again: float = fall(g)
	check(absf(again - BridgeGame.seconds_per_span(per)) < 0.1, "37: the span is walked at its own tempo")
	check(g.position == float(per), "37: and returns to the same place")

	g = two()
	fall(g)
	recover(g)
	check(series(g, "C"), "38: after the return a series lays the board")
	check(g.laid == 1 and g.target() == "G" and g.position == 0.0, "38: the run goes on")
	check(g.attempts == att - 1, "38: the lost attempt stays lost")

	g = two()
	series(g, "C")
	until_listening(g)
	fall(g)
	recover(g)
	check(not any_true(g, "C", n * 2), "39: the previous chord does not lay the next board")
	check(g.laid == 1 and g.judge.count == 0, "39: nothing counted")

	g = two()
	fall(g)
	recover(g)
	series(g, "C")
	until_listening(g)
	fall(g)
	check(g.attempts == att - 2 and g.laid == 1 and g.position == 1.0, "40: attempts are per run, not per gap")

	g = two()
	for i: int in att - 1:
		fall(g)
		recover(g)
	fall(g)
	check(
		g.state == BridgeGame.State.FALLEN and g.attempts == 0 and g.position == 1.0 and g.laid == 0,
		"41: the last attempt ends the run"
	)
	check(
		g.return_left == 0.0 and g.judge.count == 0 and g.target() == "C" and not g.listening(),
		"41: no pause, no counter, same target"
	)

	var position: float = g.position
	check(not any_true(g, "C", n * 2), "42: hear is false after FALLEN")
	g.advance(0.1)
	check(g.state == BridgeGame.State.FALLEN and g.position == position, "42: advance does nothing")
	g.start()
	check(g.state == BridgeGame.State.FALLEN, "42: start() does nothing")

	g = two()
	for i: int in att - 1:
		fall(g)
		recover(g)
	series(g, "C")
	until_listening(g)
	series(g, "G")
	for i: int in 2000:
		if g.state != BridgeGame.State.RUNNING:
			break
		g.advance(STEP)
	check(g.state == BridgeGame.State.WON and g.laid == 2, "43: the run is won after lost attempts")
	check(g.position == 3.0 and g.attempts == 1, "43: end of the bridge, one attempt left")


func view_checks() -> void:
	var att: int = BridgeGame.ATTEMPTS
	var v: BridgeView = BridgeView.new()
	check(not v.start_button.visible and v.game.state == BridgeGame.State.READY, "44: the button is hidden at first")
	v.set_listening(true)
	check(v.start_button.visible and v.start_button.text == BEGIN, "44: ready to listen shows Начать")

	v.start_button.pressed.emit()
	check(v.game.state == BridgeGame.State.RUNNING and not v.start_button.visible, "45: the button starts the run")
	check(v.attempts_label() == "Попытки: %d" % att, "45: attempts label")

	v.on_reply(reply("Em"))
	check(v.heard == "Em" and v.game.laid == 0, "46: heard shows the chord")
	v.on_reply(silence())
	check(v.heard == "", "46: silence clears heard")

	view_series(v, v.game.target())
	check(v.game.laid == 1, "47: a series of replies lays a board")

	v.begin()
	v.tick(STEP)
	check(v.game.position > 0.0, "48: tick moves the walker")
	for i: int in 1000:
		if v.game.state == BridgeGame.State.RETURNING:
			break
		v.tick(STEP)
	check(v.game.state == BridgeGame.State.RETURNING and not v.start_button.visible, "48: a fall hides the button")
	check(v.attempts_label() == "Попытки: %d" % (att - 1), "48: attempts label after a fall")

	var target: String = v.game.target()
	for i: int in ChordJudge.CONFIRM_REPLIES * 2:
		v.on_reply(reply(target))
	check(v.heard == target, "49: heard updates during the return")
	check(v.game.laid == 0 and v.game.judge.count == 0, "49: the return ignores replies")

	v.set_listening(true)
	check(not v.start_button.visible, "50: no button while returning")
	for i: int in 1000:
		if v.game.state != BridgeGame.State.RETURNING:
			break
		v.tick(STEP)
	check(v.game.state == BridgeGame.State.RUNNING and not v.start_button.visible, "50: the pause ends with no button")

	for i: int in 5000:
		if v.game.state == BridgeGame.State.FALLEN:
			break
		v.tick(STEP)
	check(v.game.state == BridgeGame.State.FALLEN, "51: the run ends")
	check(v.start_button.visible and v.start_button.text == RESTART, "51: Ещё раз appears")
	check(v.attempts_label() == "Попытки: 0", "51: no attempts left")

	v.set_listening(false)
	check(not v.start_button.visible, "52: no connection hides the button")
	v.set_listening(true)
	check(v.start_button.visible, "52: connection brings it back")

	v.start_button.pressed.emit()
	check(
		v.game.state == BridgeGame.State.RUNNING and v.game.position == 0.0 and v.game.laid == 0,
		"53: Ещё раз starts a new run"
	)
	check(v.game.attempts == att and not v.start_button.visible, "53: full attempts, no button")

	v.tick(STEP)
	var state: BridgeGame.State = v.game.state
	var position: float = v.game.position
	var laid: int = v.game.laid
	var attempts: int = v.game.attempts
	v.set_listening(false)
	check(
		v.game.state == state and v.game.position == position and v.game.laid == laid and v.game.attempts == attempts,
		"54: losing the connection leaves the game as it is"
	)
	v.free()


func compile_checks() -> void:
	for path: String in ["res://chord_judge.gd", "res://bridge_game.gd", "res://bridge_view.gd", "res://listener.gd"]:
		var script: GDScript = load(path)
		check(script != null and script.can_instantiate(), "55: %s compiles" % path)
