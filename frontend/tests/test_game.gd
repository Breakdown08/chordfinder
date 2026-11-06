extends SceneTree
## godot --headless --path frontend -s tests/test_game.gd
## Names no game class: the checks live in game_checks.gd, so a missing class is a failed check, exit 1.

const BODY: String = "res://tests/game_checks.gd"

var _done: bool = false


func _init() -> void:
	var script: GDScript = load(BODY)
	if script == null or not script.can_instantiate():
		print("FAIL game classes or checks do not compile: %s" % BODY)
		print("1 failed")
		_done = true
		quit(1)
		return
	var body: RefCounted = script.new()
	body.call("run")
	var finished: bool = body.get("finished")
	var failed: int = body.get("failed")
	if not finished:
		print("FAIL checks stopped on a runtime error")
		failed += 1
	print("%d failed" % failed)
	_done = true
	quit(1 if failed else 0)


## A runtime error aborts _init before quit(), and Godot would run forever.
func _process(_delta: float) -> bool:
	if not _done:
		print("FAIL runtime error in the test runner")
		quit(1)
	return false
