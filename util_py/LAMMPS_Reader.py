import re
import sys
import math
DEGREES_TO_RADIANS = math.pi / 180.



class LAMMPS_Reader:
    def __init__(self, state=None, nonbondFix=None, bondFix=None, angleFix=None, dihedralFix=None, improperFix=None, unitLen = 0, unitEng = 0, unitMass = 0, atomTypePrefix = '', setBounds=True):
        assert(state != None)
        self.state = state
        self.nonbondFix = nonbondFix
        self.bondFix = bondFix
        self.angleFix = angleFix
        self.dihedralFix = dihedralFix
        self.improperFix = improperFix
        assert(unitLen != 0 and unitEng != 0 and unitMass != 0)
        self.unitLen = unitLen
        self.unitEng = unitEng
        self.unitMass = unitMass
        self.myAtomTypeIds = []
        self.myAtomHandles = []
        self.atomTypePrefix = atomTypePrefix
        self.setBounds = setBounds

        self.LMPTypeToSimTypeBond = {}
        self.LMPTypeToSimTypeAngle = {}
        self.LMPTypeToSimTypeDihedral = {}
        self.LMPTypeToSimTypeImproper = {}

    def read(self, dataFn='', inputFns=[]):
        self.dataFile = open(dataFn, 'r')
        self.inputFiles = [open(inputFn, 'r') for inputFn in inputFns]
        self.allFiles = [self.dataFile ] + self.inputFiles
        self.dataFileLines = self.dataFile.readlines()
        self.inFileLines = [f.readlines() for f in self.inputFiles]
        self.allFileLines = [self.dataFileLines] + self.inFileLines
        self.isMolecular = len(self.readSection(self.dataFileLines, re.compile('Bonds'))) #this is slow, should write something to test if section exists

        self.readAtomTypes()
        self.atomIdToIdx = {}


        #might not want to change bounds if you just adding parameters to an existing simulation
        if self.setBounds:
            self.readBounds()
        self.readAtoms()
        if self.nonbondFix:
            self.readPairCoefs()
        if self.bondFix != None:
            self.readBonds()
            self.readBondCoefs()
        if self.angleFix != None:
            self.readAngles()
            self.readAngleCoefs()
        if self.dihedralFix != None:
            self.readDihedrals()
            self.readDihedralCoefs()
        if self.improperFix != None:
            self.readImpropers()
            self.readImproperCoefs()
    def isNums(self, bits):
        for b in bits:
            if b[0] == '#':
                break
            try:
                float(b)
            except ValueError:
                return False
        return len(bits)
    def emptyLine(self, line):
        bits = line.split()
        return len(bits)==0 or bits[0][0]=='#'
    def emptyLineSplit(self, bits):
        return len(bits)==0 or bits[0][0]=='#'

    def readAtomTypes(self):
        numAtomTypesRE = re.compile('^[\s]*[\d]+[\s]+atom[\s]+types')
        numTypesLines = self.scanFilesForOccurance(numAtomTypesRE, [self.dataFileLines])
        assert(len(numTypesLines) == 1)

        numTypes = int(numTypesLines[0][0])
#adding atoms with mass not set
        for i in range(numTypes):
            self.myAtomHandles.append(str(self.atomTypePrefix) + str(i))
            self.myAtomTypeIds.append(self.state.atomParams.addSpecies(self.myAtomHandles[-1], -1))

#now getting / setting masses
        masses = self.readSection(self.dataFileLines, re.compile('Mass'))
        for i, pair in enumerate(masses):
            typeIdx = self.myAtomTypeIds[i]
            mass = float(pair[1]) / self.unitMass
            self.state.atomParams.masses[typeIdx] = mass


    def readBounds(self):
        #reBase = '^\s+[\-\.\d]+\s+[\-\.\d]\s+%s\s+%s\s$'
        reBase = '^\s+[\-\.\d\e]+[\s]+[\-\.\d\e]+[\s]+%s[\s]+%s'
        bits = [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]
        lo = self.state.Vector()
        hi = self.state.Vector()
        for i, bit in enumerate(bits):
            dimRe = re.compile(reBase % bit)
            lines = self.scanFilesForOccurance(dimRe, [self.dataFileLines])
            assert(len(lines) == 1)
            lineSplit = lines[0]
            lo[i] = float(lineSplit[0]) / self.unitLen
            hi[i] = float(lineSplit[1]) / self.unitLen
        self.state.bounds.lo = lo
        self.state.bounds.hi = hi
#code SHOULD be in place to let one just change lo, hi like this.  please make sure
    def readAtoms(self):
        raw = self.readSection(self.dataFileLines, re.compile('Atoms'))
        areCharges = False  #(len(raw[0]) == 7) if self.isMolecular else (len(raw[0]) == 6)
        atomBitIdx = 2
        atomTypeIdx = 1
        chargeIdx = 0
        if self.isMolecular:
            atomBitIdx += 1
            atomTypeIdx += 1
        if (len(raw[0][atomBitIdx:]) % 3) != 0:
            areCharges = True
            chargeIdx = atomBitIdx
            atomBitIdx += 1
        for atomLine in raw:
            pos = self.state.Vector()

            pos[0] = float(atomLine[atomBitIdx]) / self.unitLen
            pos[1] = float(atomLine[atomBitIdx+1]) / self.unitLen
            pos[2] = float(atomLine[atomBitIdx+2]) / self.unitLen
            atomType = -1
            charge = 0
            if areCharges:
                charge = float(atomLine[chargeIdx])
            atomType = int(atomLine[atomTypeIdx])

            handle = self.myAtomHandles[atomType-1] #b/c lammps starts at 1
            self.atomIdToIdx[int(atomLine[0])] = len(self.state.atoms)
            self.state.addAtom(handle = handle, pos = pos, q = charge)
    def readBonds(self):
        raw = self.readSection(self.dataFileLines, re.compile('Bonds'))
        currentTypes = self.bondFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for bondLine in raw:
            bondType = int(bondLine[1])
            idA = int(bondLine[2])
            idB = int(bondLine[3])
            idxA = self.atomIdToIdx[idA]
            idxB = self.atomIdToIdx[idB]
            simType = typeOffset + bondType
            self.bondFix.createBond(self.state.atoms[idxA], self.state.atoms[idxB], type=simType)

            self.LMPTypeToSimTypeBond[bondType] = simType

    def readAngles(self):
        raw = self.readSection(self.dataFileLines, re.compile('Angles'))
        currentTypes = self.angleFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(line[1])
            ids = [int(x) for x in [line[2], line[3], line[4]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.angleFix.createAngle(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], type=simType)

            self.LMPTypeToSimTypeAngle[type] = simType

    def readDihedrals(self):
        raw = self.readSection(self.dataFileLines, re.compile('Dihedrals'))
        currentTypes = self.dihedralFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(line[1])
            ids = [int(x) for x in [line[2], line[3], line[4], line[5]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.dihedralFix.createDihedral(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], self.state.atoms[idxs[3]], type=simType)

            self.LMPTypeToSimTypeDihedral[type] = simType
    def readImpropers(self):
        raw = self.readSection(self.dataFileLines, re.compile('Impropers'))
        currentTypes = self.improperFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(line[1])
            ids = [int(x) for x in [line[2], line[3], line[4], line[5]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.improperFix.createImproper(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], self.state.atoms[idxs[3]], type=simType)

            self.LMPTypeToSimTypeImproper[type] = simType

    def readPairCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Pair Coeffs'))
        for line in rawData:
#will have to generalize this at some point
            handle = self.myAtomHandles[int(line[0]) - 1]
            eps = float(line[1]) / self.unitEng
            sig = float(line[2]) / self.unitLen
            self.nonbondFix.setParameter('sig', handleA=handle, handleB=handle, val=sig)
            self.nonbondFix.setParameter('eps', handleA=handle, handleB=handle, val=eps)
            if len(line) > 3:
                rCut = float(line[3]) / self.unitLen
                self.nonbondFix.setParameter('rCut', handleA=handle, handleB=handle, val=rCut)


        rawInput = self.scanFilesForOccurance(re.compile('pair_coeff[\s\d\-\.]+'), self.inFileLines, num=-1)
        for line in rawInput:
            handleIdxA = int(line[1]) - 1
            handleIdxB = int(line[2]) - 1
            handleA = self.myAtomHandles[handleIdxA]
            handleB = self.myAtomHandles[handleIdxB]
            eps = float(line[3]) / self.unitEng
            sig = float(line[4]) / self.unitLen
            self.nonbondFix.setParameter('sig', handleA=handleA, handleB=handleB, val=sig)
            self.nonbondFix.setParameter('eps', handleA=handleA, handleB=handleB, val=eps)
            if len(line) > 4:
                rCut = float(line[4]) / self.unitLen
                self.nonbondFix.setParameter('rCut', handleA=handleA, handleB=handleB, val=rCut)








    def readBondCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Bond Coeffs'))
        dataConverter = argumentConverters['data'][self.bondFix.type]
        inputConverter = argumentConverters['input'][self.bondFix.type]
        for line in rawData:
            args = dataConverter(self, line)
            if args != False:
                self.bondFix.setBondTypeCoefs(*args)
        rawInput = self.scanFilesForOccurance(re.compile('bond_coeff[\s\d\-\.]+'), self.inFileLines, num=-1)
        for line in rawInput:
            args = inputConverter(self, line)
            if args != False:
                self.bondFix.setBondTypeCoefs(*args)

    def readAngleCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Angle Coeffs'))
        dataConverter = argumentConverters['data'][self.angleFix.type]
        inputConverter = argumentConverters['input'][self.angleFix.type]
        for line in rawData:
            args = dataConverter(self, line)
            if args != False:
                self.angleFix.setAngleTypeCoefs(*args)
        rawInput = self.scanFilesForOccurance(re.compile('angle_coeff[\s\d\-\.]+'), self.inFileLines, num=-1)
        for line in rawInput:
            args = inputConverter(self, line)
            if args != False:
                self.angleFix.setAngleTypeCoefs(*args)


    def readDihedralCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Dihedral Coeffs'))
        dataConverter = argumentConverters['data'][self.dihedralFix.type]
        inputConverter = argumentConverters['input'][self.dihedralFix.type]
        for line in rawData:
            args = dataConverter(self, line)
            if args != False:
                self.dihedralFix.setDihedralTypeCoefs(*args)
        rawInput = self.scanFilesForOccurance(re.compile('dihedral_coeff[\s\d\-\.]+'), self.inFileLines, num=-1)
        for line in rawInput:
            args = inputConverter(self, line)
            if args != False:
                self.dihedralFix.setDihedralTypeCoefs(*args)



    def readImproperCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Improper Coeffs'))
        dataConverter = argumentConverters['data'][self.improperFix.type]
        inputConverter = argumentConverters['input'][self.improperFix.type]
        for line in rawData:
            args = dataConverter(self, line)
            if args != False:
                self.improperFix.setImproperTypeCoefs(*args)
        rawInput = self.scanFilesForOccurance(re.compile('improper_coeff[\s\d\-\.]+'), self.inFileLines, num=-1)
        for line in rawInput:
            args = inputConverter(self, line)
            if args != False:
                self.improperFix.setImproperTypeCoefs(*args)


    def stripComments(self, line):
        if '#' in line:
            return line[:line.index('#')]
        if '\n' in line:
            return line[:line.index('\n')]
        return line

    def readSection(self, dataFileLines, header):
        readData = []
        lineIdx = 0
        while lineIdx < len(dataFileLines):
            if header.search(dataFileLines[lineIdx]):
                lineIdx+=1
                break

            lineIdx+=1
        while lineIdx < len(dataFileLines) and self.emptyLine(dataFileLines[lineIdx]):
            lineIdx+=1
        while lineIdx < len(dataFileLines) and len(dataFileLines[lineIdx]):
            line = self.stripComments(dataFileLines[lineIdx])
            bits = line.split()
            if not self.emptyLineSplit(bits):
                readData.append(bits)
                lineIdx+=1
            else:
                break
        return readData

    def scanFilesForOccurance(self, regex, files, num=1):
        numOccur = 0
        fIdx = 0
        res = []
        if num==-1:
            num = sys.maxint
        while numOccur < num and fIdx < len(files):
            f = files[fIdx]
            lineNum = 0
            while numOccur < num and lineNum < len(f):
                line = f[lineNum]
                if regex.search(f[lineNum]):
                    lineStrip = self.stripComments(line)
                    bits = lineStrip.split()
                    res.append(bits)
                lineNum+=1
            fIdx+=1
        return res

#argument parsers for coefficients

def bondHarmonic_data(reader, args):
    LMPType = int(args[0])
    if not LMPType in reader.LMPTypeToSimTypeBond:
        print 'Ignoring LAMMPS bond type %d from data file.  Bond not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeBond[LMPType]
    k =  reader.unitLen * reader.unitLen * float(args[1]) / reader.unitEng
    rEq = float(args[2]) / reader.unitLen
    return [type, k, rEq]

def bondHarmonic_input(reader, args):
    LMPType = int(args[1])
    if not LMPType in reader.LMPTypeToSimTypeBond:
        print 'Ignoring LAMMPS bond type %d from input script.  Bond not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeBond[LMPType]
    k = reader.unitLen * reader.unitLen * float(args[2]) / reader.unitEng
    rEq = float(args[3]) / reader.unitLen
    return [type, k, rEq]

def angleHarmonic_data(reader, args):
    LMPType = int(args[0])
    if not LMPType in reader.LMPTypeToSimTypeAngle:
        print 'Ignoring LAMMPS angle type %d from data file.  Angle not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeAngle[LMPType]
    k = float(args[1]) / reader.unitEng
    thetaEq = float(args[2]) * DEGREES_TO_RADIANS
    return [type, k, thetaEq]

def angleHarmonic_input(reader, args):
    LMPType = int(args[1])
    if not LMPType in reader.LMPTypeToSimTypeAngle:
        print 'Ignoring LAMMPS angle type %d from input script.  Angle not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeAngle[LMPType]
    k = float(args[2]) / reader.unitEng
    thetaEq = float(args[3]) * DEGREES_TO_RADIANS
    return [type, k, thetaEq]

def dihedralOPLS_data(reader, args):
    LMPType = int(args[0])
    if not LMPType in reader.LMPTypeToSimTypeDihedral:
        print 'Ignoring LAMMPS dihedral type %d from data file.  Dihedral not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeDihedral[LMPType]
    coefs = [args[-1], args[-2], args[-3], args[-4]]
    coefs.reverse()

    coefs = [float(x) / reader.unitEng for x in coefs]
    return [type, coefs]

def dihedralOPLS_input(reader, args):
    LMPType = int(args[1])
    if not LMPType in reader.LMPTypeToSimTypeDihedral:
        print 'Ignoring LAMMPS dihedral type %d from input script.  Dihedral not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeDihedral[LMPType]
    coefs = [float(x) / reader.unitEng for x in args[2:6]]
    return [type, coefs]


def improperHarmonic_data(reader, args):
    LMPType = int(args[0])
    if not LMPType in reader.LMPTypeToSimTypeImproper:
        print 'Ignoring LAMMPS improper type %d from data file.  Improper not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeImproper[LMPType]
    k = float(args[1]) / reader.unitEng
    thetaEq = float(args[2]) * DEGREES_TO_RADIANS
    return [type, k, thetaEq]

def improperHarmonic_input(reader, args):
    LMPType = int(args[1])
    if not LMPType in reader.LMPTypeToSimTypeImproper:
        print 'Ignoring LAMMPS improper type %d from input script.  Improper not used in data file' % LMPType
        return False
    type = reader.LMPTypeToSimTypeImproper[LMPType]
    k = float(args[2]) / reader.unitEng
    thetaEq = float(args[3]) * DEGREES_TO_RADIANS
    return [type, k, thetaEq]

argumentConverters = {
        'data':
        {
            'bondHarm': bondHarmonic_data,
            'angleHarm': angleHarmonic_data,
            'dihedralOPLS': dihedralOPLS_data,
            'improperHarmonic': improperHarmonic_data
            },
        'input':
        {
            'bondHarm': bondHarmonic_input,
            'angleHarm': angleHarmonic_input,
            'dihedralOPLS': dihedralOPLS_input,
            'improperHarmonic': improperHarmonic_input
            }
        }
