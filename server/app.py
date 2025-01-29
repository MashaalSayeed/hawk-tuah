from fastapi import FastAPI, HTTPException
from schemas import ChangeStatusRequest, CommandRequest

# from drone import Drone
# mqtt_broker = "mqtt.eclipseprojects.io"
# mqtt_port = 1883
# client_id = "drone-server-1738"
# drone = Drone(mqtt_broker, mqtt_port, client_id)


app = FastAPI()
command_list = ["ARM", "TAKEOFF", "WAYPOINT", "RETURN", "LAND", "CHANGE_MODE", "VELOCITY", "DISCONNECT"]

command_queue = [
  {"action": "ARM", "arguments": {}},
  {"action": "TAKEOFF", "arguments": {"altitude": 10}},
  {"action": "WAYPOINT", "arguments": {"latitude": -35.363261, "longitude": 149.165230, "altitude": 10}},
  {"action": "RETURN", "arguments": {}},
  {"action": "DISCONNECT", "arguments": {}}
]
telemetry = {"status": "standby"}


""" USER SIDE - APIS """
@app.post("/command")
def post_command(request: CommandRequest):
    command = request.command
    arguments = request.arguments

    if command not in command_list:
        raise HTTPException(status_code=400, detail="Invalid command")
    
    command = {
        "action": command,
        "arguments": arguments
    }

    command_queue.append(command)
    return {"result": "success", "command": command}

@app.get("/command")
def get_command_list():
    return command_queue[0] if command_queue else {}

@app.get("/status")
def get_drone_status():
    return telemetry



""" DRONE SIDE - APIS """
@app.post("/telemetry")
def post_telemetry(request: ChangeStatusRequest):
    status = request.status

    telemetry["status"] = status
    print(f"Drone Status Changed to {status}")
    return {"result": "success", "status": status}

@app.get("/command/execute")
def execute_command():
    if command_queue:
        command = command_queue.pop(0)
        return {"result": "success", "command": command}
    else:
        return {"result": "success", "command": {}}