#!/usr/bin/env python3
"""
The people the robot knows (robot/people.py) - look and fix by hand until it can ask by voice.

    python3 code/people_cli.py list                 # id, name, vectors, sightings; and merge suggestions
    python3 code/people_cli.py name ID Selim
    python3 code/people_cli.py call ID Boss         # how the robot addresses them (default: their name)
    python3 code/people_cli.py merge KEEP DROP      # two ids are one person
    python3 code/people_cli.py unmerge KEEP DROP    # undo a merge
    python3 code/people_cli.py forget ID            # delete for good (face vectors + photo + voice prints)
    python3 code/people_cli.py voices               # voice prints per person: how many, how alike (low = mixed)
    python3 code/people_cli.py voices clear ID|all  # delete voice prints (the robot learns them again from
                                                    # sentences said while looking at it)
    python3 code/people_cli.py picture ID on|off|auto   # a photo on the wall (never counts as looking at the robot);
                                                        # auto = from the evidence (it never moves)

Stop alive.py first: it keeps its own copy of the database in memory and would overwrite changes.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import config                                  # noqa: E402
from robot.people import PeopleStore                      # noqa: E402
from robot.voices import VoiceStore                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", default=config.PEOPLE_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    n = sub.add_parser("name")
    n.add_argument("id")
    n.add_argument("name")
    cl = sub.add_parser("call")
    cl.add_argument("id")
    cl.add_argument("call", nargs="?", help="leave out to go back to the name")
    for c in ("merge", "unmerge"):
        m = sub.add_parser(c)
        m.add_argument("keep")
        m.add_argument("drop")
    f = sub.add_parser("forget")
    f.add_argument("id")
    vo = sub.add_parser("voices")
    vo.add_argument("action", nargs="?", choices=("clear",))
    vo.add_argument("id", nargs="?", help="a person id, or 'all'")
    ap_ = sub.add_parser("apply", help="carry out a dashboard request file (robot/commands.py) - alive.py stopped")
    ap_.add_argument("file")
    pc = sub.add_parser("picture")
    pc.add_argument("id")
    pc.add_argument("state", choices=("on", "off", "auto"))
    args = ap.parse_args()
    store = PeopleStore(args.people)

    if args.cmd == "list":
        for pid, name, nvec in store.people():
            p = store.index[pid]
            seen = datetime.fromtimestamp(p["last_seen"]).strftime("%Y-%m-%d %H:%M")
            call = f" (called {p['call']})" if p.get("call") else ""
            call += " [picture]" if store.is_picture(pid) else ""
            print(f"{pid}  {(name or '(no name)') + call:24} {nvec:3d} vectors  seen {p.get('seen', 0):4d}x, last {seen}")
        for c in store.merge_candidates():
            how = "would merge automatically" if c["auto"] else "maybe the same person"
            print(f"  {c['keep']} + {c['drop']}: {how} (sets {c['sim']:.2f}, track links {c['links']})")
        if not store.people():
            print("nobody yet")
    elif args.cmd == "name":
        store.set_name(args.id, args.name)
    elif args.cmd == "call":
        store.set_call(args.id, args.call)
    elif args.cmd == "merge":
        store.merge(args.keep, args.drop)
        VoiceStore(args.people).merge(args.keep, args.drop)
    elif args.cmd == "unmerge":
        store.unmerge(args.keep, args.drop)
        print("note: voice prints are not split back (they do not say whose they were) - if needed: "
              f"people_cli.py voices clear {args.keep} (they are learnt again)")
    elif args.cmd == "picture":
        store.set_picture(args.id, {"on": True, "off": False, "auto": None}[args.state])
    elif args.cmd == "voices":
        voices = VoiceStore(args.people)
        if args.action == "clear":
            if not args.id:
                sys.exit("voices clear: give a person id or 'all'")
            ids = list(voices.prints) if args.id == "all" else [args.id]
            for pid in ids:
                voices.forget(pid)
            print(f"voice prints deleted: {', '.join(ids) or 'none'}")
        else:
            for pid in sorted(voices.prints):
                c = voices.consistency(pid)
                alike = f"alike {c.mean():.2f} (lowest {c.min():.2f})" if len(c) else "alike -"
                print(f"{pid}  {(store.name(pid) or '(no name)'):16} {voices.count(pid):3d} prints  {alike}")
            if not voices.prints:
                print("no voice prints")
    elif args.cmd == "apply":
        from robot.commands import PeopleCommands, apply_file
        apply_file(args.file, PeopleCommands(store, VoiceStore(args.people)))
    elif args.cmd == "forget":
        from robot.commands import forget_recorded
        store.forget(args.id)
        VoiceStore(args.people).forget(args.id)
        print(f"recorded face pictures deleted: {forget_recorded(args.id)}")


if __name__ == "__main__":
    main()
