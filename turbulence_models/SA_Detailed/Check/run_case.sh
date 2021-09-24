# Run potentialFoam for initial condition
cd Setup
potentialFoam -writep >> init_condition_setup.txt
cd ..

cp Setup/0/U 0/U
cp Setup/0/p 0/p

simpleFoam >> training_log.txt
postProcess -latestTime -func writeCellCentres
postProcess -latestTime -func sampleDict
