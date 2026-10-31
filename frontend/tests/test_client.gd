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

	# Strict typing warnings are errors: a script that breaks them does not compile.
	for path: String in ["res://listener.gd", "res://frames.gd", "res://reply.gd"]:
		var script: GDScript = load(path)
		check(script != null and script.can_instantiate(), "compiles: " + path)

	print("%d failed" % _failed)
	quit(1 if _failed else 0)
