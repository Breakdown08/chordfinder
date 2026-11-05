extends SceneTree
## godot --headless --path frontend -s tests/test_client.gd
## A dozen asserts do not need GUT/gdUnit4; exit code is the verdict.

var _failed: int = 0


func check(ok: bool, what: String) -> void:
	print(("ok   " if ok else "FAIL ") + what)
	if not ok:
		_failed += 1


func _init() -> void:
	var s: PackedFloat32Array = PackedFloat32Array([0.0, 1.0, -1.0, 0.123456789, -3.4e38, 1e-45])
	var bytes: PackedByteArray = Frames.pack(s)
	check(bytes.size() == s.size() * 4, "pack: 4 bytes per sample")
	check(Frames.unpack(bytes) == s, "pack -> unpack is lossless")
	check(bytes.slice(4, 8) == PackedByteArray([0x00, 0x00, 0x80, 0x3f]), "pack: little-endian 1.0f")

	var frames: PackedVector2Array = PackedVector2Array([Vector2(1, 0), Vector2(0.5, 0.5), Vector2(-1, 1)])
	check(Frames.mono(frames) == PackedFloat32Array([0.5, 0.5, 0.0]), "mono: average of channels")

	# Accumulation: N capture chunks -> exactly N * size samples to send.
	var pending: PackedFloat32Array = PackedFloat32Array()
	var chunk: PackedVector2Array = PackedVector2Array()
	chunk.resize(512)
	for i: int in 37:
		pending.append_array(Frames.mono(chunk))
	check(pending.size() == 37 * 512, "accumulate: N chunks -> N * size samples")
	check(Frames.pack(pending).size() == 37 * 512 * 4, "accumulate: frame bytes match")

	check(not Frames.has_sound(PackedFloat32Array([0.0, 0.0, -0.0])), "has_sound: zeros are no sound")
	check(Frames.has_sound(PackedFloat32Array([0.0, 1e-7])), "has_sound: any non-zero sample")

	var r: Reply = Reply.parse('{"chord": "Am", "confidence": 0.91, "rms": 0.03, "stable": null, "samples": 6000}')
	check(r.chord == "Am" and r.stable == "" and r.samples == 6000 and r.error == "", "reply: chord, null stable")
	check(is_equal_approx(r.confidence, 0.91) and is_equal_approx(r.rms, 0.03), "reply: numbers")
	r = Reply.parse('{"chord": null, "confidence": null, "rms": 0.001, "stable": null, "samples": 12000}')
	check(r.chord == "" and r.confidence == 0.0, "reply: silence")
	check(Reply.parse('{"error": "frame length 3 is not a multiple of 4"}').error.begins_with("frame"), "reply: error")
	check(Reply.parse("[1, 2]").error != "", "reply: not an object is an error")

	r = Reply.parse('{"chord": "C", "confidence": 0.7, "rms": 0.02, "probabilities": {"Am": 0.3, "C": 0.7}}')
	check(r.probabilities.keys() == ["Am", "C"], "reply: probabilities keep the server order")
	check(is_equal_approx(r.probabilities["C"], 0.7), "reply: probability value")
	check(Reply.parse('{"probabilities": null}').probabilities.is_empty(), "reply: silence has no probabilities")

	check(Bars.level(0.0) == 0.0, "level: digital silence is the bottom")
	check(Bars.level(1.0) == 1.0, "level: full scale is the top")
	check(Bars.level(0.1) > Bars.level(0.01) and Bars.level(0.01) > Bars.level(0.001), "level: louder is higher")

	var m: Measure = Measure.new()
	check(m.chord() == "Am" and m.state == Measure.State.WAITING, "measure: waits for the first chord")
	m.heard(2, 0.0)
	check(m.strums == 0, "measure: strums before the button do not count")
	m.record()
	m.heard(2, 1.0)
	check(m.strums == 2 and not m.tail_over(9.0), "measure: two strums are not a take")
	check(m.counter() == "Am  2/3", "measure: big counter")
	m.heard(3, 10.0)
	check(not m.tail_over(11.0), "measure: the last strum rings out on the record")
	check(m.tail_over(10.0 + Measure.TAIL_SECONDS), "measure: then the take ends")
	m.advance()
	check(m.chord() == "Bb" and m.state == Measure.State.WAITING and m.strums == 0, "measure: next chord")
	m.redo()
	check(m.chord() == "Am" and m.state == Measure.State.WAITING, "measure: redo between takes goes back")
	m.record()
	m.heard(1, 1.0)
	m.redo()
	check(m.chord() == "Am" and m.strums == 0 and m.state == Measure.State.WAITING, "measure: redo drops the take")
	m.index = Measure.CHORDS.size() - 1
	m.record()
	m.advance()
	check(m.state == Measure.State.DONE and m.text().contains("Готово"), "measure: done after the last chord")
	r = Reply.parse('{"chord": "Am", "take": "Am", "strums": 2}')
	check(r.take == "Am" and r.strums == 2, "reply: take and strums")

	# Strict typing warnings are errors: a script that breaks them does not compile.
	for path: String in ["res://listener.gd", "res://frames.gd", "res://reply.gd", "res://bars.gd", "res://measure.gd"]:
		var script: GDScript = load(path)
		check(script != null and script.can_instantiate(), "compiles: " + path)

	print("%d failed" % _failed)
	quit(1 if _failed else 0)
