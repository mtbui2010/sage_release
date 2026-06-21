# pyplanner/few_shot_examples.py
#
# Hand-crafted few-shot examples aligned with the current STEP_SCHEMA.
# Each example covers: task → intent analysis → evaluated plan.
#
# Rules encoded (from STEP_SCHEMA):
#   - MoveTo <room>     before any Find in that room
#   - Find <object>     before Pick / Open / Close / TurnOn / TurnOff
#   - Pick              only after Find; object field is ignored at runtime
#   - Place <recep>     only after Holding + MoveTo <recep>
#   - Implicit containers: Find <container> → Open → Find <object> → Pick
#                          → Find <container> → Close
#
# Evaluation legend used in comments:
#   ✓ intent coverage   — every object/location in task appears in plan
#   ✓ preconditions     — Find before Pick, Arrived before Place
#   ✓ container rule    — Open/Close wrap objects inside containers

FEW_SHOT_EXAMPLES = '''
=== EXAMPLE 1 ===
Task: Place apple in to dish in dining table

Intent objects & locations: Apple, Dish (Plate/Bowl), DiningTable

Reasoning:
- Apple is on a surface in the current room; no container → no Open needed.
- Dish (Plate) is on the DiningTable → move to DiningTable, place there.
- Sequence: find apple → pick → navigate to dining table → place.
- Evaluation: ✓ intent coverage (Apple, DiningTable), ✓ preconditions, ✓ no container.

Plan:
{"steps": [
  {"action": "Find",   "object": "Apple"},
  {"action": "Pick",   "object": "Apple"},
  {"action": "MoveTo", "object": "DiningTable"},
  {"action": "Place",  "object": "DiningTable"}
]}

=== EXAMPLE 2 ===
Task: Grab remote control on dining table in kitchen and place on sofa in living room

Intent objects & locations: RemoteControl, Kitchen, DiningTable, LivingRoom, Sofa

Reasoning:
- RemoteControl is on DiningTable in Kitchen → navigate to Kitchen first, then DiningTable.
- Destination is Sofa in LivingRoom → navigate to LivingRoom then Sofa.
- No container involved → no Open/Close needed.
- Evaluation: ✓ intent coverage (RemoteControl, Kitchen, DiningTable, LivingRoom, Sofa),
              ✓ preconditions (Find before Pick, MoveTo Sofa before Place).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "DiningTable"},
  {"action": "Find",   "object": "RemoteControl"},
  {"action": "Pick",   "object": "RemoteControl"},
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "Sofa"},
  {"action": "Place",  "object": "Sofa"}
]}

=== EXAMPLE 3 ===
Task: Bring the towel from the kitchen table. Place it on the dresser in front of the bathroom.

Intent objects & locations: Towel, Kitchen, DiningTable (kitchen table), Bathroom, Dresser

Reasoning:
- Towel is on the kitchen table (DiningTable) → move to Kitchen, then DiningTable, find Towel.
- Destination is Dresser in Bathroom → move to Bathroom, then Dresser, place.
- No container involved.
- Evaluation: ✓ intent coverage (Towel, Kitchen, DiningTable, Bathroom, Dresser),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "DiningTable"},
  {"action": "Find",   "object": "Towel"},
  {"action": "Pick",   "object": "Towel"},
  {"action": "MoveTo", "object": "Bathroom"},
  {"action": "MoveTo", "object": "Dresser"},
  {"action": "Place",  "object": "Dresser"}
]}

=== EXAMPLE 4 ===
Task: Put the cup from the dining table into the sink

Intent objects & locations: Cup (Mug), DiningTable, Sink

Reasoning:
- Cup is on the DiningTable → navigate there and pick it up.
- Sink is in the Kitchen → navigate to Kitchen, then Sink, place cup.
- No container involved.
- Evaluation: ✓ intent coverage (Mug, DiningTable, Sink), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "DiningTable"},
  {"action": "Find",   "object": "Mug"},
  {"action": "Pick",   "object": "Mug"},
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "Sink"},
  {"action": "Place",  "object": "Sink"}
]}

=== EXAMPLE 5 ===
Task: Could you bring the remote control from the dining table? Place it on the table next in the living room.

Intent objects & locations: RemoteControl, DiningTable, LivingRoom, CoffeeTable (table in living room)

Reasoning:
- RemoteControl is on DiningTable (no room specified → assume current room).
- Destination: table in LivingRoom → CoffeeTable.
- Sequence: navigate DiningTable → find → pick → navigate LivingRoom → navigate CoffeeTable → place.
- Evaluation: ✓ intent coverage (RemoteControl, DiningTable, LivingRoom, CoffeeTable),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "DiningTable"},
  {"action": "Find",   "object": "RemoteControl"},
  {"action": "Pick",   "object": "RemoteControl"},
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "CoffeeTable"},
  {"action": "Place",  "object": "CoffeeTable"}
]}

=== EXAMPLE 6 ===
Task: Throw the can from the table in the living room into the recycling bin in the kitchen

Intent objects & locations: SodaCan, LivingRoom, CoffeeTable, Kitchen, GarbageCan (recycling bin)

Reasoning:
- Can is on CoffeeTable in LivingRoom → navigate there and pick it up.
- Recycling bin (GarbageCan) is in Kitchen → navigate to Kitchen then GarbageCan, place.
- No container involved.
- Evaluation: ✓ intent coverage (SodaCan, LivingRoom, CoffeeTable, Kitchen, GarbageCan),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "CoffeeTable"},
  {"action": "Find",   "object": "SodaCan"},
  {"action": "Pick",   "object": "SodaCan"},
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "GarbageCan"},
  {"action": "Place",  "object": "GarbageCan"}
]}

=== EXAMPLE 7 ===
Task: Turn light on

Intent objects & locations: FloorLamp (or DeskLamp / LightSwitch)

Reasoning:
- No navigation specified → assume light is in current room.
- Find the lamp, then TurnOn.
- No container, no pick/place needed.
- Evaluation: ✓ intent coverage (FloorLamp), ✓ preconditions (Find before TurnOn).

Plan:
{"steps": [
  {"action": "Find",   "object": "FloorLamp"},
  {"action": "TurnOn", "object": "FloorLamp"}
]}

=== EXAMPLE 8 ===
Task: Get the milk from the fridge and put it on the kitchen counter

Intent objects & locations: Milk, Fridge, Kitchen, CounterTop

Reasoning:
- Milk is inside the Fridge → implicit container rule: Find Fridge → Open → Find Milk → Pick → Find Fridge → Close.
- Destination is CounterTop in Kitchen → MoveTo CounterTop → Place CounterTop.
- Evaluation: ✓ intent coverage (Milk, Fridge, Kitchen, CounterTop),
              ✓ preconditions, ✓ container rule (Fridge opened and closed).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Open",   "object": "Fridge"},
  {"action": "Find",   "object": "Milk"},
  {"action": "Pick",   "object": "Milk"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Close",  "object": "Fridge"},
  {"action": "MoveTo", "object": "CounterTop"},
  {"action": "Place",  "object": "CounterTop"}
]}

=== EXAMPLE 9 ===
Task: Put bread in the toaster and turn it on

Intent objects & locations: Bread, Toaster, Kitchen

Reasoning:
- Bread is on the CounterTop (no container) → Find Bread → Pick.
- Toaster is on the CounterTop → MoveTo Toaster, Place Toaster, then TurnOn.
- Toaster is a receptacle: no Open/Close needed for the toaster slot itself.
- Evaluation: ✓ intent coverage (Bread, Toaster, Kitchen), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Bread"},
  {"action": "Pick",   "object": "Bread"},
  {"action": "MoveTo", "object": "Toaster"},
  {"action": "Place",  "object": "Toaster"},
  {"action": "Find",   "object": "Toaster"},
  {"action": "TurnOn", "object": "Toaster"}
]}

=== EXAMPLE 10 ===
Task: Get the soap from the bathroom cabinet and put it next to the sink

Intent objects & locations: SoapBottle, Bathroom, Cabinet, Sink

Reasoning:
- SoapBottle is inside a Cabinet → implicit container rule: Find Cabinet → Open → Find SoapBottle → Pick → Find Cabinet → Close.
- Destination is beside the Sink → MoveTo Sink → Place Sink.
- Evaluation: ✓ intent coverage (SoapBottle, Bathroom, Cabinet, Sink),
              ✓ preconditions, ✓ container rule (Cabinet opened and closed).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Bathroom"},
  {"action": "Find",   "object": "Cabinet"},
  {"action": "Open",   "object": "Cabinet"},
  {"action": "Find",   "object": "SoapBottle"},
  {"action": "Pick",   "object": "SoapBottle"},
  {"action": "Find",   "object": "Cabinet"},
  {"action": "Close",  "object": "Cabinet"},
  {"action": "MoveTo", "object": "Sink"},
  {"action": "Place",  "object": "Sink"}
]}

=== EXAMPLE 11 ===
Task: Heat the leftover food from the fridge in the microwave

Intent objects & locations: Plate (food), Fridge, Kitchen, Microwave

Reasoning:
- Plate is inside the Fridge → container rule: Open Fridge, pick Plate, Close Fridge.
- Microwave is also a container → must Open Microwave, Place inside, Close, then TurnOn.
- Two-container pattern: Fridge (source) and Microwave (destination).
- Evaluation: ✓ intent coverage (Plate, Fridge, Kitchen, Microwave),
              ✓ preconditions, ✓ container rule on both Fridge and Microwave.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Open",   "object": "Fridge"},
  {"action": "Find",   "object": "Plate"},
  {"action": "Pick",   "object": "Plate"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Close",  "object": "Fridge"},
  {"action": "MoveTo", "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "Open",   "object": "Microwave"},
  {"action": "Place",  "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "Close",  "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "TurnOn", "object": "Microwave"}
]}

=== EXAMPLE 12 ===
Task: Put the dirty mug from the coffee table into the kitchen sink

Intent objects & locations: Mug, LivingRoom, CoffeeTable, Kitchen, Sink

Reasoning:
- Mug is on CoffeeTable in LivingRoom → navigate there, find and pick up.
- Sink is in Kitchen → navigate Kitchen, then Sink, place.
- No container involved.
- Evaluation: ✓ intent coverage (Mug, LivingRoom, CoffeeTable, Kitchen, Sink),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "CoffeeTable"},
  {"action": "Find",   "object": "Mug"},
  {"action": "Pick",   "object": "Mug"},
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "Sink"},
  {"action": "Place",  "object": "Sink"}
]}

=== EXAMPLE 13 ===
Task: Bring the book from the bookshelf in the living room to the nightstand in the bedroom

Intent objects & locations: Book, LivingRoom, BookShelf, Bedroom, NightStand

Reasoning:
- Book is on BookShelf in LivingRoom → navigate there, find and pick up.
- Destination: NightStand in Bedroom → navigate Bedroom then NightStand, place.
- BookShelf is not a closed container → no Open/Close needed.
- Evaluation: ✓ intent coverage (Book, LivingRoom, BookShelf, Bedroom, NightStand),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "BookShelf"},
  {"action": "Find",   "object": "Book"},
  {"action": "Pick",   "object": "Book"},
  {"action": "MoveTo", "object": "Bedroom"},
  {"action": "MoveTo", "object": "NightStand"},
  {"action": "Place",  "object": "NightStand"}
]}

=== EXAMPLE 14 ===
Task: Get bread from the kitchen cabinet, put it in the toaster and turn it on

Intent objects & locations: Bread, Kitchen, Cabinet, Toaster

Reasoning:
- Bread is inside a Cabinet → container rule: Find Cabinet → Open → Find Bread → Pick → Find Cabinet → Close.
- Toaster is on CounterTop → MoveTo Toaster, Place, then TurnOn.
- Evaluation: ✓ intent coverage (Bread, Kitchen, Cabinet, Toaster),
              ✓ preconditions, ✓ container rule (Cabinet opened and closed).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Cabinet"},
  {"action": "Open",   "object": "Cabinet"},
  {"action": "Find",   "object": "Bread"},
  {"action": "Pick",   "object": "Bread"},
  {"action": "Find",   "object": "Cabinet"},
  {"action": "Close",  "object": "Cabinet"},
  {"action": "MoveTo", "object": "Toaster"},
  {"action": "Place",  "object": "Toaster"},
  {"action": "Find",   "object": "Toaster"},
  {"action": "TurnOn", "object": "Toaster"}
]}

=== EXAMPLE 15 ===
Task: Turn on the desk lamp in the bedroom

Intent objects & locations: DeskLamp, Bedroom

Reasoning:
- DeskLamp is in the Bedroom → navigate there first.
- Find lamp, TurnOn. No pick/place needed.
- Evaluation: ✓ intent coverage (DeskLamp, Bedroom), ✓ preconditions (Find before TurnOn).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Bedroom"},
  {"action": "Find",   "object": "DeskLamp"},
  {"action": "TurnOn", "object": "DeskLamp"}
]}

=== EXAMPLE 16 ===
Task: Bring the laptop from the bedroom to the coffee table in the living room

Intent objects & locations: Laptop, Bedroom, LivingRoom, CoffeeTable

Reasoning:
- Laptop is in Bedroom, possibly on NightStand or Desk → navigate Bedroom, find and pick.
- Destination: CoffeeTable in LivingRoom → navigate LivingRoom then CoffeeTable, place.
- No container involved.
- Evaluation: ✓ intent coverage (Laptop, Bedroom, LivingRoom, CoffeeTable),
              ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Bedroom"},
  {"action": "Find",   "object": "Laptop"},
  {"action": "Pick",   "object": "Laptop"},
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "CoffeeTable"},
  {"action": "Place",  "object": "CoffeeTable"}
]}

=== EXAMPLE 17 ===
Task: Put the vase from the bookshelf on the coffee table

Intent objects & locations: Vase, BookShelf, LivingRoom, CoffeeTable

Reasoning:
- Vase is on BookShelf in LivingRoom → navigate LivingRoom, MoveTo BookShelf, find and pick.
- Destination: CoffeeTable (same room) → MoveTo CoffeeTable, place.
- No container involved.
- Evaluation: ✓ intent coverage (Vase, BookShelf, LivingRoom, CoffeeTable), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "BookShelf"},
  {"action": "Find",   "object": "Vase"},
  {"action": "Pick",   "object": "Vase"},
  {"action": "MoveTo", "object": "CoffeeTable"},
  {"action": "Place",  "object": "CoffeeTable"}
]}

=== EXAMPLE 18 ===
Task: Get a towel from the bathroom and bring it to the kitchen counter

Intent objects & locations: Towel, Bathroom, Kitchen, CounterTop

Reasoning:
- Towel is in Bathroom (on a rack or shelf, not in a closed container) → navigate Bathroom, find and pick.
- Destination: CounterTop in Kitchen → navigate Kitchen then CounterTop, place.
- Evaluation: ✓ intent coverage (Towel, Bathroom, Kitchen, CounterTop), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Bathroom"},
  {"action": "Find",   "object": "Towel"},
  {"action": "Pick",   "object": "Towel"},
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "CounterTop"},
  {"action": "Place",  "object": "CounterTop"}
]}

=== EXAMPLE 19 ===
Task: Make coffee — place the mug under the coffee machine and start it

Intent objects & locations: Mug, CoffeeMachine, Kitchen, CounterTop

Reasoning:
- Mug is on CounterTop in Kitchen → navigate Kitchen, find and pick Mug.
- CoffeeMachine is on CounterTop → MoveTo CoffeeMachine, place Mug, then TurnOn.
- No container involved.
- Evaluation: ✓ intent coverage (Mug, CoffeeMachine, Kitchen), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Mug"},
  {"action": "Pick",   "object": "Mug"},
  {"action": "MoveTo", "object": "CoffeeMachine"},
  {"action": "Place",  "object": "CoffeeMachine"},
  {"action": "Find",   "object": "CoffeeMachine"},
  {"action": "TurnOn", "object": "CoffeeMachine"}
]}

=== EXAMPLE 20 ===
Task: Put the egg from the fridge into the microwave and heat it up

Intent objects & locations: Egg, Fridge, Kitchen, Microwave

Reasoning:
- Egg is inside the Fridge → container rule: Open Fridge, pick Egg, Close Fridge.
- Microwave is also a container → Open Microwave, Place, Close, TurnOn.
- Two containers: apply container rule to both.
- Evaluation: ✓ intent coverage (Egg, Fridge, Kitchen, Microwave),
              ✓ preconditions, ✓ container rule on both Fridge and Microwave.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Open",   "object": "Fridge"},
  {"action": "Find",   "object": "Egg"},
  {"action": "Pick",   "object": "Egg"},
  {"action": "Find",   "object": "Fridge"},
  {"action": "Close",  "object": "Fridge"},
  {"action": "MoveTo", "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "Open",   "object": "Microwave"},
  {"action": "Place",  "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "Close",  "object": "Microwave"},
  {"action": "Find",   "object": "Microwave"},
  {"action": "TurnOn", "object": "Microwave"}
]}

=== EXAMPLE 21 ===
Task: Take the remote control from the sofa and put it in the drawer of the TV stand

Intent objects & locations: RemoteControl, LivingRoom, Sofa, Drawer (TVStand)

Reasoning:
- RemoteControl is on/near Sofa in LivingRoom → navigate LivingRoom, MoveTo Sofa, find and pick.
- Destination is inside a Drawer (container) → Find Drawer → Open → Place Drawer → Find Drawer → Close.
- Evaluation: ✓ intent coverage (RemoteControl, LivingRoom, Sofa, Drawer),
              ✓ preconditions, ✓ container rule (Drawer opened and closed).

Plan:
{"steps": [
  {"action": "MoveTo", "object": "LivingRoom"},
  {"action": "MoveTo", "object": "Sofa"},
  {"action": "Find",   "object": "RemoteControl"},
  {"action": "Pick",   "object": "RemoteControl"},
  {"action": "Find",   "object": "Drawer"},
  {"action": "Open",   "object": "Drawer"},
  {"action": "Place",  "object": "Drawer"},
  {"action": "Find",   "object": "Drawer"},
  {"action": "Close",  "object": "Drawer"}
]}

=== EXAMPLE 22 ===
Task: Wash the apple from the kitchen counter in the sink

Intent objects & locations: Apple, Kitchen, CounterTop, Sink

Reasoning:
- Apple is on CounterTop in Kitchen → navigate Kitchen, MoveTo CounterTop, find and pick.
- Sink is in Kitchen (same room) → MoveTo Sink, turn on faucet, wash apple, turn off.
- Evaluation: ✓ intent coverage (Apple, Kitchen, CounterTop, Sink), ✓ preconditions.

Plan:
{"steps": [
  {"action": "MoveTo", "object": "Kitchen"},
  {"action": "MoveTo", "object": "CounterTop"},
  {"action": "Find",   "object": "Apple"},
  {"action": "Pick",   "object": "Apple"},
  {"action": "MoveTo", "object": "Sink"},
  {"action": "Find",   "object": "Faucet"},
  {"action": "TurnOn", "object": "Faucet"},
  {"action": "Wash",   "object": "Apple"},
  {"action": "Find",   "object": "Faucet"},
  {"action": "TurnOff","object": "Faucet"}
]}
'''
