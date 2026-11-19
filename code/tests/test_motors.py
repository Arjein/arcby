import unittest

from robot import config
from robot.motors import Motors, to_pwm
from robot.state import State
from tests.fakes import FakeClock, FakeDriver, imu_reading

DT = 1.0 / config.DRIVE_RATE_HZ


class MotorsTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.driver = FakeDriver()
        self.m = Motors(self.state, self.driver, clock=self.clock)
        self.safe()

    def safe(self, stop=False, reasons=()):
        self.state.publish("safety", {"stop": stop, "reasons": list(reasons), "latched": []})

    def run_for(self, seconds, left=None, right=None, keep_safe=True):
        """Step the loop; re-send the request every step like a live sender."""
        out = None
        for _ in range(int(round(seconds / DT))):
            self.clock.sleep(DT)
            if keep_safe:
                self.safe()
            if left is not None:
                self.m.request(left, right, "test")
            out = self.m.step()
        return out

    def test_dead_band_mapping(self):
        self.assertEqual(to_pwm(0.0), 0.0)
        self.assertEqual(to_pwm(0.005), 0.0)
        self.assertAlmostEqual(to_pwm(0.01), config.DRIVE_MIN + (config.DRIVE_MAX - config.DRIVE_MIN) * 0.01)
        self.assertAlmostEqual(to_pwm(1.0), config.DRIVE_MAX)
        self.assertAlmostEqual(to_pwm(-1.0), -config.DRIVE_MAX)
        self.assertAlmostEqual(to_pwm(5.0), config.DRIVE_MAX)

    def test_not_a_number_is_stop_not_full_speed(self):                       # review 2026-11-01
        nan = float("nan")
        self.m.request(nan, float("inf"), "test")
        self.assertEqual((self.m._req.left, self.m._req.right), (0.0, 0.0))
        self.m.request_velocity(nan, nan, "test")
        self.assertEqual((self.m._req.v, self.m._req.turn), (0.0, 0.0))

    def test_no_command_means_no_motion(self):
        self.run_for(0.5)
        self.assertEqual(self.driver.out, (0.0, 0.0))
        self.assertEqual(self.state.get("motors")["reason"], "no command yet")

    def test_soft_start_limits_the_rise(self):
        target = to_pwm(0.5)
        out = self.run_for(DT, 0.5, 0.5)
        self.assertAlmostEqual(out[0], config.DRIVE_ACCEL * DT)
        out = self.run_for(0.1, 0.5, 0.5)
        self.assertAlmostEqual(out[0], config.DRIVE_ACCEL * 0.12, places=6)
        out = self.run_for(target / config.DRIVE_ACCEL, 0.5, 0.5)
        self.assertAlmostEqual(out[0], target)
        for (l, r), (l2, r2) in zip(self.driver.history, self.driver.history[1:]):
            self.assertLessEqual(abs(l2 - l), config.DRIVE_ACCEL * DT + 1e-9)

    def test_slowing_down_is_faster_than_speeding_up(self):
        self.run_for(1.0, 1.0, 1.0)
        self.assertAlmostEqual(self.driver.out[0], config.DRIVE_MAX)
        out = self.run_for(DT, 0.0, 0.0)
        self.assertAlmostEqual(out[0], config.DRIVE_MAX - config.DRIVE_DECEL * DT)
        out = self.run_for(config.DRIVE_MAX / config.DRIVE_DECEL, 0.0, 0.0)
        self.assertEqual(out, (0.0, 0.0))

    def test_watchdog_stops_a_silent_sender(self):
        self.run_for(1.0, 0.5, 0.5)
        self.assertGreater(self.driver.out[0], 0)
        self.run_for(config.DRIVE_CMD_TIMEOUT_S - 0.05)      # no new requests, still within timeout
        self.assertGreater(self.driver.out[0], 0)
        self.run_for(0.1 + to_pwm(0.5) / config.DRIVE_DECEL)
        self.assertEqual(self.driver.out, (0.0, 0.0))
        self.assertEqual(self.state.get("motors")["reason"], "command timeout")

    def test_safety_stop_is_instant(self):
        self.run_for(1.0, 1.0, 1.0)
        self.safe(stop=True, reasons=["tipped"])
        self.clock.sleep(DT)
        self.m.request(1.0, 1.0, "test")
        self.assertEqual(self.m.step(), (0.0, 0.0))
        self.assertEqual(self.driver.out, (0.0, 0.0))
        self.assertEqual(self.state.get("motors")["reason"], "safety: tipped")
        self.assertFalse(self.state.get("motors")["moving"])

    def test_missing_or_old_safety_verdict_stops(self):
        self.run_for(1.0, 1.0, 1.0)
        out = self.run_for(config.SAFETY_MAX_AGE_S + 2 * DT, 1.0, 1.0, keep_safe=False)
        self.assertEqual(out, (0.0, 0.0))
        self.assertEqual(self.state.get("motors")["reason"], "safety not running")

    def test_after_a_safety_stop_it_soft_starts_again(self):
        self.run_for(1.0, 1.0, 1.0)
        self.safe(stop=True, reasons=["bump"])
        self.clock.sleep(DT); self.m.request(1.0, 1.0); self.m.step()
        out = self.run_for(DT, 1.0, 1.0)
        self.assertAlmostEqual(out[0], config.DRIVE_ACCEL * DT)

    def test_settled_time_for_bump_blanking(self):
        self.run_for(0.1, 0.5, 0.5)
        m = self.state.get("motors")
        self.assertTrue(m["moving"])
        self.assertIsNone(m["settled_since"])            # still ramping
        self.run_for(0.5, 0.5, 0.5)
        settled = self.state.get("motors")["settled_since"]
        self.assertIsNotNone(settled)
        self.run_for(0.2, 0.5, 0.5)
        self.assertEqual(self.state.get("motors")["settled_since"], settled)
        self.run_for(DT, 0.8, 0.8)                        # new target: ramping again
        self.assertIsNone(self.state.get("motors")["settled_since"])

    def test_heading_hold_steers_back(self):
        self.state.publish("imu", imu_reading(self.clock(), yaw=0.0))
        self.run_for(DT, 0.5, 0.5)
        self.assertEqual(self.state.get("motors")["hold_yaw"], 0.0)
        self.run_for(1.0, 0.5, 0.5)
        # drifted 5 deg to the RIGHT (yaw -5): right wheel must get more
        for _ in range(5):
            self.state.publish("imu", imu_reading(self.clock(), yaw=-5.0))
            out = self.run_for(DT, 0.5, 0.5)
        self.assertAlmostEqual(out[1] - out[0], 2 * config.HEADING_KP * 5.0)
        # drifted far LEFT: left wheel more, limited to HEADING_MAX_STEER
        self.state.publish("imu", imu_reading(self.clock(), yaw=40.0))
        out = self.run_for(DT, 0.5, 0.5)
        self.assertAlmostEqual(out[0] - out[1], 2 * config.HEADING_MAX_STEER)

    def test_heading_hold_only_when_straight(self):
        self.state.publish("imu", imu_reading(self.clock(), yaw=0.0))
        self.run_for(1.0, -0.2, 0.2)
        self.assertIsNone(self.state.get("motors")["hold_yaw"])
        self.state.publish("imu", imu_reading(self.clock(), yaw=30.0))
        out = self.run_for(DT, 0.5, 0.5)
        self.assertEqual(self.state.get("motors")["hold_yaw"], 30.0)   # holds the NEW heading

    def test_heading_hold_works_backward(self):
        self.state.publish("imu", imu_reading(self.clock(), yaw=0.0))
        self.run_for(1.0, -0.5, -0.5)
        self.state.publish("imu", imu_reading(self.clock(), yaw=-5.0))   # turned right
        out = self.run_for(DT, -0.5, -0.5)
        # yaw rate ~ (right - left): to turn left again right - left must be > 0
        self.assertGreater(out[1] - out[0], 0)

    def test_run_loop_stops_motors_on_exit(self):
        import threading
        stop = threading.Event()
        clock_state = State()
        clock_state.publish("safety", {"stop": False, "reasons": [], "latched": []})
        drv = FakeDriver()
        m = Motors(clock_state, drv)
        th = threading.Thread(target=m.run, args=(stop,))
        th.start()
        stop.set()
        th.join(timeout=1)
        self.assertFalse(th.is_alive())
        self.assertGreaterEqual(drv.stops, 1)
        self.assertEqual(drv.out, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()


class Plant:
    """Tiny robot model for closed-loop tests: wheels do nothing below `start` PWM,
    turn rate = k * (right - left) with a lag, right wheel `right_gain` as strong."""

    def __init__(self, k=400.0, start=0.30, right_gain=1.0, tau=0.1, cm_s=100.0, offset=False):
        self.k, self.start, self.right_gain, self.tau, self.cm_s = k, start, right_gain, tau, cm_s
        self.offset = offset                # True: wheel speed ~ (PWM - start) instead of PWM
        self.yaw = self.rate = 0.0
        self.forward_cm = 0.0               # how far the centre moved forward (ignores heading)

    def update(self, out, dt):
        if self.offset:
            l, r = (0.0 if abs(v) < self.start else (abs(v) - self.start) * (1 if v > 0 else -1)
                    for v in out)
        else:
            l, r = (0.0 if abs(v) < self.start else v for v in out)
        target = self.k * (r * self.right_gain - l)
        self.rate += (target - self.rate) * min(1.0, dt / self.tau)
        self.yaw += self.rate * dt
        self.forward_cm += self.cm_s * (l + r * self.right_gain) / 2 * dt


class SpinPlant(Plant):
    """Both wheels opposite, like the carpet measurement 2026-10-29: stuck below 0.24 PWM,
    ~250 deg/s at 0.26, +590 deg/s per PWM above."""

    def update(self, out, dt):
        l, r = out
        mag = (abs(l) + abs(r)) / 2
        if l * r < 0 and mag >= 0.24:
            target = (250.0 + 590.0 * max(0.0, mag - 0.26)) * (1 if r > l else -1)
        else:
            Plant.update(self, out, dt)                 # pivots / straight: the simple model
            return
        self.rate += (target - self.rate) * min(1.0, dt / self.tau)
        self.yaw += self.rate * dt


class VelocityTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.driver = FakeDriver()
        self.m = Motors(self.state, self.driver, clock=self.clock)

    def run_plant(self, seconds, v, turn, plant):
        rates, outs = [], []
        for _ in range(int(round(seconds / DT))):
            self.clock.sleep(DT)
            self.state.publish("safety", {"stop": False, "reasons": [], "latched": []})
            self.state.publish("imu", imu_reading(self.clock(), yaw=plant.yaw, yaw_rate=plant.rate))
            self.m.request_velocity(v, turn, "test")
            out = self.m.step()
            plant.update(out, DT)
            rates.append(plant.rate)
            outs.append(out)
        return rates, outs

    def settled_rate(self, rates, last_s=1.0):
        tail = rates[-int(last_s / DT):]
        return sum(tail) / len(tail)

    def test_curve_while_driving_reaches_the_asked_rate(self):
        plant = Plant()
        rates, _ = self.run_plant(3.0, 0.4, -60.0, plant)            # forward, turning right 60 deg/s
        self.assertAlmostEqual(self.settled_rate(rates), -60.0, delta=5)

    def test_rate_is_reached_even_if_the_model_is_wrong(self):
        for k in (250.0, 600.0):                                   # carpet vs tiles, battery level...
            plant = Plant(k=k)
            self.setUp()
            rates, _ = self.run_plant(3.0, 0.4, 60.0, plant)
            self.assertAlmostEqual(self.settled_rate(rates), 60.0, delta=6, msg=f"k={k}")

    def mode(self, name):
        old = config.TURN_ON_SPOT
        config.TURN_ON_SPOT = name
        self.addCleanup(setattr, config, "TURN_ON_SPOT", old)

    def pivot_mode(self):
        self.mode("pivot")

    def test_default_always_spins_on_both_wheels(self):
        """Chosen 2026-10-29: on the spot always both wheels - a slow request turns at the spin minimum."""
        self.assertEqual(config.TURN_ON_SPOT, "spin")
        plant = SpinPlant()
        rates, outs = self.run_plant(2.0, 0.0, 90.0, plant)
        self.assertAlmostEqual(self.settled_rate(rates, 0.5), config.SPIN_MIN_DPS, delta=15)
        for l, r in outs[-25:]:
            self.assertLess(l, 0)
            self.assertGreater(r, 0)
        self.assertLess(abs(plant.forward_cm), 1.0)

    def test_fast_turn_on_the_spot_uses_both_wheels(self):
        """Selim 2026-10-29: turn with both wheels at once (about the centre) - possible from ~250 deg/s."""
        plant = SpinPlant()
        rates, outs = self.run_plant(2.0, 0.0, 290.0, plant)
        self.assertAlmostEqual(self.settled_rate(rates, 0.5), 290.0, delta=10)
        for l, r in outs[-25:]:
            self.assertLess(l, 0)                                  # left turn: left back,
            self.assertGreater(r, 0)                               # right forward
            self.assertGreaterEqual(abs(l), config.SPIN_START)
        self.assertLess(abs(plant.forward_cm), 1.0)                # turns about the centre

    def test_spin_power_comes_from_the_measured_curve(self):
        plant = SpinPlant()
        rates, outs = self.run_plant(1.0, 0.0, -300.0, plant)
        l, r = outs[-1]
        self.assertAlmostEqual(l, 0.26 + 50 / 590, delta=0.03)     # power ~ the measured curve
        self.assertAlmostEqual(self.settled_rate(rates, 0.3), -300.0, delta=10)
        self.assertEqual(r, -l)

    def test_auto_slow_turn_on_the_spot_pivots(self):
        self.mode("auto")
        plant = SpinPlant()
        rates, outs = self.run_plant(3.0, 0.0, 90.0, plant)
        self.assertAlmostEqual(self.settled_rate(rates), 90.0, delta=10)
        self.assertTrue(all(0.0 in o for o in outs[-25:]))         # one wheel stands
        self.assertLess(abs(plant.forward_cm), 8.0)

    def test_auto_switch_has_a_gap(self):
        self.mode("auto")
        plant = SpinPlant()
        self.run_plant(1.0, 0.0, 300.0, plant)
        self.assertTrue(self.m._spinning)
        self.run_plant(0.5, 0.0, 255.0, plant)                     # between DOWN and UP: keep spinning
        self.assertTrue(self.m._spinning)
        self.run_plant(0.5, 0.0, 200.0, plant)
        self.assertFalse(self.m._spinning)
        self.run_plant(0.5, 0.0, 260.0, plant)                     # between: keep pivoting
        self.assertFalse(self.m._spinning)

    def test_releasing_a_spin_never_turns_the_other_way(self):
        """Selim 2026-10-29: hold A, release -> it spun RIGHT for a moment. The keys send turn -0.0
        on release and its minus sign picked the direction. Both kinds of zero, both directions."""
        for turn, zero in ((90.0, -0.0), (90.0, 0.0), (-90.0, 0.0), (-90.0, -0.0)):
            self.setUp()
            plant = SpinPlant()
            self.run_plant(1.0, 0.0, turn, plant)
            _, outs = self.run_plant(0.5, 0.0, zero, plant)
            for l, r in outs:
                if turn > 0:
                    self.assertFalse(l > 0 and r < 0, f"right spin after releasing a LEFT turn: {(l, r)}")
                else:
                    self.assertFalse(l < 0 and r > 0, f"left spin after releasing a RIGHT turn: {(l, r)}")
            self.assertEqual(outs[0], (0.0, 0.0))                       # stops at once

    def test_a_spin_starts_softly(self):
        """2026-11-08 power cut right after a spin: from standstill the spin PWM rises over SPIN_RAMP_S, not at once
        (the PI part alone asked ~0.5 at the first step)."""
        plant = SpinPlant()
        _, outs = self.run_plant(1.0, 0.0, 250.0, plant)
        steps = int(round(config.SPIN_RAMP_S / DT))
        mags = [abs(o[1]) for o in outs]
        self.assertLessEqual(mags[0], config.SPIN_START / config.SPIN_RAMP_S * DT + 1e-9)
        for i in range(steps - 1):
            self.assertLessEqual(mags[i], mags[i + 1] + 1e-9)
            self.assertLess(mags[i], config.SPIN_START)
        self.assertGreaterEqual(mags[-1], config.SPIN_START)        # then the normal spin

    def test_a_reversed_spin_starts_softly_again(self):
        plant = SpinPlant()
        self.run_plant(0.6, 0.0, 300.0, plant)
        _, outs = self.run_plant(0.6, 0.0, -300.0, plant)
        first = next(o for o in outs if o[0] > 0)                   # the first step of the right spin
        self.assertLess(first[0], config.SPIN_START / 2)

    def test_spin_ramp_zero_is_the_old_hard_start(self):
        old = config.SPIN_RAMP_S
        config.SPIN_RAMP_S = 0.0
        self.addCleanup(setattr, config, "SPIN_RAMP_S", old)
        _, outs = self.run_plant(0.1, 0.0, 250.0, SpinPlant())
        self.assertGreaterEqual(abs(outs[0][1]), config.SPIN_START)

    def test_spin_never_reverses_the_wheels_while_turning(self):
        """Black box 20:14: the controller flipped the wheels every 40 ms during a slow request
        (it overshot, then reversed to correct) - 1 g jolts. Now one direction per request."""
        plant = SpinPlant()
        _, outs = self.run_plant(3.0, 0.0, 90.0, plant)
        self.assertTrue(all(l < 0 < r for l, r in outs[3:]), "a left spin must keep left < 0 < right")

    def test_spin_left_right_does_not_creep(self):
        plant = SpinPlant()
        for i in range(8):
            self.run_plant(0.6, 0.0, 300.0 if i % 2 == 0 else -300.0, plant)
        self.assertLess(abs(plant.forward_cm), 1.0)

    def test_slow_turn_on_the_spot_pivots_on_one_wheel(self):
        self.pivot_mode()
        plant = Plant()
        rates, outs = self.run_plant(3.0, 0.0, 40.0, plant)          # standing, left 40 deg/s
        self.assertAlmostEqual(self.settled_rate(rates), 40.0, delta=6)
        for l, r in outs[-25:]:                                    # left turn on ONE wheel:
            self.assertTrue((l == 0.0 and r > 0) or (r == 0.0 and l < 0), (l, r))  # right fwd or left back

    def test_left_right_left_right_does_not_creep_forward(self):
        """Selim 2026-10-29: A-D-A-D on the spot crept forward (every pivot drove a wheel forward)."""
        self.pivot_mode()
        plant = Plant()
        for i in range(8):
            self.run_plant(0.4, 0.0, 90.0 if i % 2 == 0 else -90.0, plant)
        self.assertLess(abs(plant.forward_cm), 8.0)
        self.assertGreater(min(o[0] for o in self.driver.history), -1)   # sanity

    def test_long_slow_turn_stays_on_the_spot(self):
        self.pivot_mode()
        plant = Plant()
        rates, outs = self.run_plant(4.0, 0.0, 60.0, plant)
        self.assertLess(abs(plant.forward_cm), 8.0)
        self.assertAlmostEqual(self.settled_rate(rates), 60.0, delta=10)
        self.assertTrue(any(o[0] < 0 for o in outs) and any(o[1] > 0 for o in outs))  # both pivot kinds

    def test_fast_turn_on_the_spot_spins_on_both_wheels(self):
        plant = Plant()
        rates, outs = self.run_plant(3.0, 0.0, -280.0, plant)
        self.assertAlmostEqual(self.settled_rate(rates), -280.0, delta=15)
        l, r = outs[-1]
        self.assertGreater(l, 0)                                   # right spin: left fwd, right back
        self.assertLess(r, 0)

    def test_straight_holds_the_heading_with_a_weak_wheel(self):
        plant = Plant(right_gain=0.9)
        self.run_plant(4.0, 0.5, 0.0, plant)
        self.assertLess(abs(plant.yaw), 3.0)                         # open loop would drift ~90 deg
        self.assertIsNotNone(self.state.get("motors")["hold_yaw"])

    def test_a_forward_start_gives_the_right_wheel_a_little_more_at_first(self):
        old = config.DRIVE_START_RIGHT
        config.DRIVE_START_RIGHT = 0.06
        self.addCleanup(setattr, config, "DRIVE_START_RIGHT", old)
        plant = Plant()
        _, outs = self.run_plant(0.1, 0.1, 0.0, plant)
        left, right = outs[-1]
        self.assertAlmostEqual(right - left, config.DRIVE_START_RIGHT, delta=0.03)
        _, outs = self.run_plant(config.DRIVE_START_S + 0.5, 0.1, 0.0, plant)
        left, right = outs[-1]
        self.assertLess(abs(right - left), 0.03)                     # then only the heading hold
        self.m.stop()
        self.run_plant(0.0, 0.0, 0.0, plant)
        _, outs = self.run_plant(0.1, -0.1, 0.0, plant)                # backing: no bias
        self.assertLess(abs(outs[-1][1] - outs[-1][0]), 0.03)

    def test_turn_rate_changes_smoothly(self):
        plant = Plant()
        self.run_plant(DT, 0.4, 300.0, plant)
        self.assertLessEqual(self.state.get("motors")["turn_goal"], config.TURN_ACCEL_DPS2 * DT + 1e-9)

    def test_letting_go_of_steer_holds_the_new_heading(self):
        plant = Plant()
        self.run_plant(1.5, 0.4, 90.0, plant)
        self.run_plant(2.0, 0.4, 0.0, plant)
        yaw_after = plant.yaw
        self.run_plant(2.0, 0.4, 0.0, plant)
        self.assertLess(abs(plant.yaw - yaw_after), 2.0)

    def test_velocity_requests_also_time_out(self):
        plant = Plant()
        self.run_plant(1.0, 0.5, 0.0, plant)
        for _ in range(int((config.DRIVE_CMD_TIMEOUT_S + 0.3) / DT)):
            self.clock.sleep(DT)
            self.state.publish("safety", {"stop": False, "reasons": [], "latched": []})
            out = self.m.step()
        self.assertEqual(out, (0.0, 0.0))

    def test_turn_rate_request_is_capped(self):
        self.m.request_velocity(0.0, 5000.0)
        self.assertEqual(self.m._req.turn, config.TURN_MAX_DPS)


class GpioDriverTest(unittest.TestCase):
    """The real driver class on gpiozero mock pins; the pinctrl call is intercepted (no real pins)."""

    def setUp(self):
        from gpiozero.pins.mock import MockFactory, MockPWMPin
        from unittest import mock
        from robot import motors
        self.run = mock.patch.object(motors.subprocess, "run").start()
        self.addCleanup(mock.patch.stopall)
        self.d = motors.GpioDriver(pin_factory=MockFactory(pin_class=MockPWMPin))
        self.L, self.R = self.d.motors

    def pins(self, m):
        return (round(m.forward_device.value, 3), round(m.backward_device.value, 3), m.enable_device.value)

    def test_uses_the_pins_from_pinout(self):
        names = [d.pin.info.name for m in self.d.motors
                 for d in (m.forward_device, m.backward_device, m.enable_device)]
        self.assertEqual(names, ["GPIO27", "GPIO22", "GPIO17", "GPIO23", "GPIO24", "GPIO25"])

    def test_directions_and_speeds(self):
        self.d.set(0.5, -0.3)
        self.assertEqual(self.pins(self.L), (0.5, 0.0, 1))         # left forward
        self.assertEqual(self.pins(self.R), (0.0, 0.3, 1))         # right backward
        self.d.set(0.0, 2.0)                                      # 0 = stop, >1 capped
        self.assertEqual(self.pins(self.L), (0.0, 0.0, 1))
        self.assertEqual(self.pins(self.R), (1.0, 0.0, 1))

    def test_stop(self):
        self.d.set(0.7, 0.7)
        self.d.stop()
        self.assertEqual(self.pins(self.L)[:2], (0.0, 0.0))
        self.assertEqual(self.pins(self.R)[:2], (0.0, 0.0))

    def test_close_stops_and_pulls_the_pins_down(self):
        self.d.set(0.7, 0.7)
        self.d.close()
        args = self.run.call_args[0][0]
        self.assertEqual(args, ["pinctrl", "set", "17,27,22,23,24,25", "ip", "pd"])
        self.assertTrue(all(d.closed for m in self.d.motors
                            for d in (m.forward_device, m.backward_device, m.enable_device)))


class MotorsMoreTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = State(clock=self.clock)
        self.driver = FakeDriver()
        self.m = Motors(self.state, self.driver, clock=self.clock)

    def safe(self, stop=False):
        self.state.publish("safety", {"stop": stop, "reasons": ["test"] if stop else [], "latched": []})

    def drive(self, seconds, v=None, turn=0.0, wheels=None):
        out = None
        for _ in range(int(round(seconds / DT))):
            self.clock.sleep(DT)
            self.safe()
            if v is not None:
                self.m.request_velocity(v, turn, "t")
            if wheels is not None:
                self.m.request(*wheels, "t")
            out = self.m.step()
        return out

    def test_stop_request(self):
        self.drive(1.0, wheels=(0.5, 0.5))
        self.m.stop("t")
        self.clock.sleep(DT)
        self.safe()
        self.m.step()
        self.assertEqual(self.state.get("motors")["target"], (0.0, 0.0))

    def test_close_closes_the_driver(self):
        self.m.close()
        self.assertTrue(self.driver.closed)

    def test_velocity_mode_safety_stop_is_instant_and_restart_is_soft(self):
        self.drive(1.0, v=0.6)
        self.assertGreater(self.driver.out[0], 0.4)
        self.clock.sleep(DT)
        self.safe(stop=True)
        self.m.request_velocity(0.6, 0.0)
        self.assertEqual(self.m.step(), (0.0, 0.0))
        self.assertEqual(self.m._turn_i, 0.0)                     # nothing left over
        out = self.drive(DT, v=0.6)
        self.assertLessEqual(abs(out[0]), config.DRIVE_ACCEL * DT + config.HEADING_MAX_STEER + 1e-9)

    def test_switching_between_wheel_and_velocity_requests(self):
        self.drive(0.5, wheels=(0.5, 0.5))
        out = self.drive(1.0, v=-0.5)
        self.assertLess(out[0], 0)
        out = self.drive(1.0, wheels=(0.0, 0.0))
        self.assertEqual(out, (0.0, 0.0))

    def test_moving_flag_for_turn_only_requests(self):
        self.drive(0.5, v=0.0, turn=100.0)
        self.assertTrue(self.state.get("motors")["moving"])        # turning on the spot counts as moving

    def test_run_loop_survives_falling_behind(self):
        import threading
        stop = threading.Event()
        calls = []

        class SlowClock:
            t = 0.0
            def __call__(self):
                self.t += 0.05                                   # every read: 50 ms later (behind)
                calls.append(self.t)
                if len(calls) > 40:
                    stop.set()
                return self.t
        m = Motors(self.state, self.driver, clock=SlowClock())
        self.safe()
        m.run(stop)                                              # must return, not spin forever
        self.assertTrue(stop.is_set())
        self.assertEqual(self.driver.out, (0.0, 0.0))
