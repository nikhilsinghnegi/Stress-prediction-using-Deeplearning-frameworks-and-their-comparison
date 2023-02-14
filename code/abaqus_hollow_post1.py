from odbAccess import *

data_set = 20

for kk in range(data_set+1):

    mm = str(kk+1)

    opname='New_Master_von_' + mm
    opfiles = 'D:\\new_place_for_odb_file\\' + opname + '.odb'

    odb = openOdb(opfiles)
    outputFile = open('loadDisp_'+opname+'.dat','w')

    # lastFrame = odb.steps['AnonymousSTEP1'].frame[-1]

    stress = odb.steps['AnonymousSTEP1'].frames[-1].fieldOutputs['S']
    disp = odb.steps['AnonymousSTEP1'].frames[-1].fieldOutputs['U']

    nodes = odb.rootAssembly.instances['PART-1-1'].nodeSets['NODE'].nodes
    node = odb.rootAssembly.instances['PART-1-1'].nodeSets['NODE']
    
    elements = odb.rootAssembly.instances['PART-1-1'].elements
    # elements_2 = odb.rootAssembly.instances['PART-1-1'].elementSets['SET2'].elements

    for jj in range(len(nodes)):

        stress_Val_1 = stress.values[jj].data[0]
        stress_Val_2 = stress.values[jj].data[1]
        stress_Val_3 = stress.values[jj].data[2]
        stress_Val_4 = stress.values[jj].data[3]

        disp_Val_1 = disp.values[jj].data[0]
        disp_Val_2 = disp.values[jj].data[1]

        print(disp_Val_1, disp_Val_2, stress_Val_1, stress_Val_2, stress_Val_3, stress_Val_4)

        outputFile.write('%10.4E %10.4E %10.4E %10.4E %10.4E %10.4E \n' %(disp_Val_1,disp_Val_2,stress_Val_1,stress_Val_2,stress_Val_3,stress_Val_4))

    outputFile.close( )

    coord_name = 'node_coord_' + mm + '.dat'
    outputFile_1 = open(coord_name,'w')

    for ff in range(len(nodes)):

        x_coord = nodes[ff].coordinates[0]
        y_coord = nodes[ff].coordinates[1]
        z_coord = nodes[ff].coordinates[2]

        print(x_coord, y_coord, z_coord)

        outputFile_1.write('%10.4E %10.4E %10.4E \n' %(x_coord, y_coord, z_coord))

    outputFile_1.close()
    
    element_name = 'element_indice_' + mm + '.dat'
    outputFile_2 = open(element_name,'w')
    
    for hh in range(len(elements)):
    
        element_id = hh + 1
        element_1 = elements[hh].connectivity[0]
        element_2 = elements[hh].connectivity[1]
        element_3 = elements[hh].connectivity[2]
        
        print(element_id, element_1, element_2, element_3)
        
        outputFile_2.write('%10.4E %10.4E %10.4E %10.4E \n' %(element_id, element_1, element_2, element_3))
        
    outputFile_2.close()