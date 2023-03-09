import os
import subprocess


def runSumo(flow, ratio, setup, filename, out_dir, use_gui=False):
    '''
    Runs SUMO sim with the given set of inputs, and setup file
    Inputs
        carPerHour      - car flow
        truckPerHour    - truck flow
        eTruckPerHour   - empty truck flow
        stopsPerHour    - demand of the road/edge that we are simulating
        setup           - dictionary type containing simulation parameters

    Outputs
        Outputs 'filename.csv' for the sim into the defined directory out_dir
        Temp files needed to run sim are generated in out_dir/temp/
    '''
    #---------------------------------------------------------#
    #                 INITIALIZATION
    #---------------------------------------------------------#
    mph2mps = setup['mph2mps']
    sigma = setup['sigma']
    startTime = setup['startTime']
    endTime = setup['endTime']
    speedLimit = setup['speedLimit']
    roadLength = setup['roadLength']
    numLanes = setup['numLanes']
    numStops = setup['numStops']
    carLength = setup['carLength']
    carSpeed = setup['carSpeed']
    carAccel = setup['carAccel']
    carDecel = setup['carDecel']
    truckLength = setup['truckLength']
    truckSpeed = setup['truckSpeed']
    truckAccel = setup['truckAccel']
    truckDecel = setup['truckDecel']
    stopTime = setup['stopTime']

    # Flow/model paremters
    carProb = flow*(1-ratio)/3600
    truckProb = flow*ratio/3600

    # create required dirs
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir+'temp/')
    if not os.path.exists(out_dir+'temp/'):
        os.makedirs(out_dir+'temp/')
    tempDir = out_dir + 'temp/'
    #---------------------------------------------------------#
    #                 SIMULATION SETUP
    #---------------------------------------------------------#
    # Generate nodes xml file:
    f1 = open(tempDir+'truck-capacity.nod.xml', 'w')
    f1.write('<nodes>' + '\n')
    f1.write('\t' + '<node id="start" x="0" y="0"/>' + '\n')
    f1.write('\t' + '<node id="end" x="' + str(roadLength) + '" y="0"/>' + '\n')
    f1.write('</nodes>')
    f1.close()

    # Generate edges xml file:
    f2 = open(tempDir+'truck-capacity.edg.xml', 'w')
    f2.write('<edges>' + '\n')
    f2.write('\t' + '<edge id="road" from="start" to="end" speedLimit="' + str(speedLimit) +
             '" numLanes="' + str(numLanes) + '"/>' + '\n')
    f2.write('</edges>')
    f2.close()

    # Generate network xml file from nodes and edges xml files (utilizes subprocess to suppress command line output):
    subprocess.check_output(["netconvert", "--node-files="+tempDir+"truck-capacity.nod.xml", 
                                           "--edge-files="+tempDir+"truck-capacity.edg.xml",
                                           "--output-file="+tempDir+"truck-capacity.net.xml"])
    # Generate routes xml file:
    f3 = open(tempDir+'truck-capacity.rou.xml', 'w')
    f3.write('<routes>' + '\n')
    f3.write('\t' + '<vType id="car" guiShape="passenger" length="' + str(carLength) + '" maxSpeed="' + str(carSpeed) +
             '" accel="' + str(carAccel) + '" decel="' + str(carDecel) + '" sigma="' + str(sigma) + '"/>' + '\n')
    f3.write('\t' + '<vType id="truck" guiShape="truck" length="' + str(truckLength) + '" maxSpeed="' + str(truckSpeed) +
             '" accel="' + str(truckAccel) + '" decel="' + str(truckDecel) + '" sigma="' + str(sigma) + '"/>' + '\n')
    for i in range(numLanes):
        f3.write('\t' + '<flow id="carFlow' + str(i) + '" type="car" color="yellow" via="road" departLane="' + str(i)+ 
                '" begin="' + str(startTime) + '" end="' + str(endTime) + '" probability="' + str(round(carProb/numLanes,5)) + '"/>' + '\n')
    # NEW empty Truck
    # f3.write('\t' + '<flow id="etruckFlow" type="truck" color="blue" via="road" departLane="free" ' +
    #          'begin="' + str(startTime) +
    #          '" end="' + str(endTime) + '" probability="' + str(eTruckProb) + '"/>' + '\n')

    for i in range(1, numStops + 1):
        f3.write('\t' + '<flow id="truckFlow' + str(i) + '" type="truck" color="red" via="road" departLane="free" begin="' + str(startTime) +
                 '" end="' + str(endTime) + '" probability="' + str(truckProb/numStops) + '">' + '\n')
        f3.write('\t' + '\t' + '<stop busStop="truckStop' + str(i) + '" duration="' + str(stopTime) + '"/>' + '\n')
        f3.write('\t' + '</flow>' + '\n')
    f3.write('</routes>')
    f3.close()

    # Generate additional elements (bus stops) xml file:
    f4 = open(tempDir+'truck-capacity.stops.xml', 'w')
    f4.write('<additional>' + '\n')
    for i in range(1, numStops + 1):
        f4.write('\t' + '<busStop id="truckStop' + str(i) + '" lane="road_0" ' +
                 'startPos="' + str(i * roadLength / (numStops + 1) - truckLength / 2) +
                 '" endPos="' + str(i * roadLength / (numStops + 1) + truckLength / 2) + '"/>' + '\n')
    f4.write('</additional>')
    f4.close()

    # Generate SUMO configuration file:
    f5 = open(tempDir+'truck-capacity.sumocfg', 'w')
    f5.write('<configuration>' + '\n')
    f5.write('\t' + '<input>' + '\n')
    f5.write('\t' + '\t' + '<net-file value="truck-capacity.net.xml"/>' + '\n')
    f5.write('\t' + '\t' + '<route-files value="truck-capacity.rou.xml"/>' + '\n')
    f5.write('\t' + '\t' + '<additional-files value="truck-capacity.stops.xml"/>' + '\n')
    f5.write('\t' + '</input>' + '\n')
    f5.write('</configuration>')
    f5.close()
    #---------------------------------------------------------#
    #                 SIMULATION 
    #---------------------------------------------------------#
    # Run simulation from cmd line (utilizes subprocess to suppress command line output):
    subprocess.check_output(["sumo", "-c",tempDir+"truck-capacity.sumocfg", "-e4000", "--tripinfo-output", out_dir+filename+".xml"])
    subprocess.check_output(["python", "/home/mark/Documents/code/drone/sumo/utils/xml2csv.py",out_dir+filename+".xml"])

    if use_gui:
        subprocess.check_output(["sumo-gui", "-c",tempDir+"truck-capacity.sumocfg", "-e5000"])
