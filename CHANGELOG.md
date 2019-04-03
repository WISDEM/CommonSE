# CommonSE Changelog

## 0.2.0 ([04/01/2019])

[Garrett Barter](mailto: garrett.barter@nrel.gov)

- OpenMDAO1 release

## 0.1.0 (June 27, 2014)

Andrew Ning <andrew.ning@nrel.gov>

- initial release

## 0.1.1 ([11/03/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[NEW]:

- added rna file with ability to aggregate rna mass properties and process rotor loads for TowerSE and JacketSE (formally functionality was in TowerSE)

[FIX]:

- error in csmPPI relative file path fixed

# 0.1.2 ([02/07/2015])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[CHANGE]:

- minor updates to new Material.py file

[FIX]:

- Jacobian for wave component fixed

- Slight update for rna code to match tower changes

- wave height 1.1 multiplier removed since this should be supplied via hs input

# 0.1.3 ([07/08/2015])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[NEW]:

- Added embedment checks for jacket structure, modified Tube to calculate pseudomass, added G to default material parameters
