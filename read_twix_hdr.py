import numpy as np
import re,os,operator
from itertools import chain
from functools import reduce

def read_twix_hdr(f):
    # function to read raw data header information from siemens MRI scanners
    # (currently VB and VD software versions are supported and tested).
    #
    # Author: Philipp Ehses (philipp.ehses@tuebingen.mpg.de), Mar/11/2014

    nbuffers = np.fromfile(f,dtype=np.uint32,count=1)[0]

    prot = {}
    for b in range(nbuffers):
        # now read string up to null termination
        bufname = np.fromfile(f,dtype=np.uint8,count=10)
        bufname = ''.join(map(chr,bufname))
        bufname = re.search('^\w*',bufname).group(0)
        f.seek(len(bufname) - 9,os.SEEK_CUR)
        buflen = np.fromfile(f,dtype=np.uint32,count=1)[0]
        # print(buflen)
        buffer = np.fromfile(f,dtype=np.uint8,count=buflen)
        buffer = ''.join(map(chr,buffer)).replace('\n\s*\n','') # delete empty lines
        prot[bufname] = parse_buffer(buffer)

        rstraj = None
        if ('Meas' in prot) and ('alRegridMode' in prot['Meas']) and (prot['Meas']['alRegridMode'][0] > 1):
            raise NotImplementedError()

            ncol = prot['Meas']['alRegridDestSamples'][0]
            dwelltime = prot['Meas']['aflRegridADCDuration'][0]/ncol
            gr_adc = np.zeros(ncol,dtype='single')
            start = prot['Meas']['alRegridRampupTime'][0] - (prot['Meas']['aflRegridADCDuration'][0] - prot['Meas']['alRegridFlattopTime'][0])/2
            time_adc = start + dwelltime*np.arange(0,5,ncol)
            ixUp = time_adc <= prot['Meas']['alRegridRampupTime'][0]
            ixFlat = (time_adc <= prot['Meas']['alRegridRampupTime'][0] + prot['Meas']['alRegridFlattopTime'][0]) & ~ixUp
            ixDn = ~ixUp & ~ixFlat
            gr_adc[ixFlat] = 1

            if prot['Meas']['alRegridMode'][0] == 2:
                # trapezoidal gradient
                gr_adc[ixUp] = time_adc[ixUp]/prot['Meas']['alRegridRampupTime'][0]
                gr_adc[ixDn] = 1 - (time_adc[ixDn] - prot['Meas']['alRegridRampupTime'][0] - prot['Meas']['alRegridFlattopTime'][0])/prot['Meas']['alRegridRampdownTime'][0]
            elif prot['Meas']['alRegridMode'][0] == 4:
                # sinusoidal gradient
                gr_adc[ixUp] = np.sin(np.pi/2*time_adc[ixUp]/prot['Meas']['alRegridRampupTime'][0])
                gr_adc[ixDn] = np.sin(np.pi/2*(1+(time_adc[ixDn] - prot['Meas']['alRegridRampupTime'][0] - prot['Meas']['alRegridFlattopTime'][0])/prot['Meas']['alRegridRampdownTime'][0]))
            else:
                raise ValueError('regridding mode unknown')

            # make sure that gr_adc is always positive (rstraj needs to be
            # strictly monotonic):
            gr_adc = np.max([ gr_adc,1e-4 ])
            rstraj = (np.cumsum(gr_adc.flatten()) - ncol/2)/np.sum(gr_adc.flatten())
            rstraj = rstraj - rstraj[int(ncol/2)]

            # scale rstraj by kmax (only works if all slices have same FoV!!!)
            kmax = prot['MeasYaps']['sKSpace']['lBaseResolution']/['prot']['MeasYaps']['sSliceArray']['asSlice'][1]['dReadoutFOV']
            rstraj = kmax*rstraj

    return(prot,rstraj)


def parse_buffer(buffer):
    idx0 = buffer.find('### ASCCONV BEGIN')
    idx1 = buffer.find('### ASCCONV END ###')
    if idx0 > 0:
        ascconv = buffer[idx0:idx1]
        xprot = buffer.replace(ascconv,'')
    else:
        ascconv = ''
        xprot = buffer

    if ascconv != '':
        prot = parse_ascconv(ascconv);
    else:
        prot = {}

    if xprot != '':
        xprot = parse_xprot(xprot)
        if type(xprot) is {}:
            # Merge the two dictionaries
            prot = { **xprot,**prot }

    return(prot)


def parse_xprot(buffer):
    xprot = {}
    tokens = re.finditer('<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)',buffer)
    double_tokens = re.finditer('<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)',buffer)

    # Concatenate all tokens
    all_tokens = chain(tokens,double_tokens)

    for tok in all_tokens:
        name = tok.group(1)

        # field name has to start with letter
        if not name[0].isalnum():
            name = 'x' + name

        # The value should be the last token
        value = tok.groups()[-1]

        # See if it's an array
        arr = value.split()
        if len(arr) == 1:
            if arr[0].isnumeric():
                xprot[name] = float(arr[0])
            else:
                xprot[name] = arr[0]
        elif len(arr) > 1:
            if all(el.isnumeric() for el in arr):
                xprot[name] = [ float(x) for x in value.split() ]
            else:
                xprot[name] = value
        else:
            xprot[name] = value


    # print(json.dumps(xprot,indent=2))
    return(xprot)


def parse_ascconv(buffer):
    mrprot = {}
    vararray = re.finditer('(?P<name>\S*)\s*=\s*(?P<value>\S*)',buffer)

    for var in vararray:

        # Make the value a number if we can
        if var['value'].isnumeric():
            value = float(var['value'])
        else:
            value = var['value']

        # Get ready for the multi-level dictionary
        name = var['name'].replace('[','.').replace(']','').split('.')
        if len(name) > 1:
            for ii,key in enumerate(name[:-1]):
                if key not in reduce(operator.getitem,name[:ii],mrprot):
                    reduce(operator.getitem,name[:ii],mrprot)[key] = {}
            reduce(operator.getitem,name[:-1],mrprot)[name[-1]] = value
        else:
            mrprot[name[0]] = value

    return(mrprot)
