import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
import glob
import logging
import os,json
from time import time,sleep
from io import StringIO
from twix_map_obj import twix_map_obj
from read_twix_hdr import read_twix_hdr
from tqdm import tqdm,trange
from itertools import chain

logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.DEBUG)

def mapVBVD(filename=None,**kwargs):

    if filename is None:
        # If user doens't supply anything, pull up a file selction GUI
        root = Tk()
        root.withdraw()
        filename = Path(askopenfilename(parent=root,title='Please select binary file to read',filetypes = (('Siemens raw data',"*.dat"),)))
    else:
        if type(filename) is str:
            # assume that complete path is given
            filename = Path(filename)

            # Make sure we have the right file extension
            if filename.suffix != '.dat':
                filename = Path('%s.dat' % filename)  # adds filetype ending to file
                logging.info('File extension ".dat" not found, appending to end of filename.')

        elif type(filename) is int:
            # filename not a string, so assume that it is the MeasID
            measID = filename
            filelist = dir('*.dat')
            filesfound = 0

            # for file in glob.glob('./^meas_MID0*%d_*.dat'):
            files =  glob.glob('meas_MID*%d*.dat' % measID)
            if len(files) == 0:
                raise ValueError('File with meas. id %d not found.' % measID)
            elif len(files) > 1:
                logging.warning('Multiple files with meas. id %d found. Choosing first occurence.' % measID)

            filename = Path(files[0])

    # add absolute path, when no path is given
    filename = filename.resolve()

    ###### Parse kwargs ######
    arg = {}
    arg['bReadImaScan'] = True
    arg['bReadNoiseScan'] = True
    arg['bReadPCScan'] = True
    arg['bReadRefScan'] = True
    arg['bReadRefPCScan'] = True
    arg['bReadRTfeedback'] = True
    arg['bReadPhaseStab'] = True
    arg['bReadHeader'] = True

    k = 0
    keys = list(kwargs.keys())
    while k < len(kwargs):
        if type(kwargs[keys[k]]) is not str:
            raise ValueError('string expected')

        if key.lower() in [ 'readheader','readhdr','header','hdr' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.bReadHeader = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.bReadHeader = True
                k += 1

        elif key.lower() in [ 'removeos','rmos' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.removeOS = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.removeOS = True
                k += 1

        elif key.lower() in [ 'doaverage','doave','ave','average' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.doAverage = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.doAverage = True
                k += 1

        elif key.lower() in [ 'averagereps','averagerepetitions' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.averageReps = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.averageReps = True
                k += 1

        elif key.lower() in [ 'averagesets' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.averageSets = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.averageSets = True
                k += 1

        elif key.lower() in [ 'ignseg','ignsegments','ignoreseg','ignoresegments' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.ignoreSeg = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.ignoreSeg = True
                k += 1

        elif key.lower() in [ 'rampsampregrid','regrid' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.rampSampRegrid = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.rampSampRegrid = True
                k += 1

        elif key.lower() in [ 'rawdatacorrect','dorawdatacorrect' ]:
            if (len(kwargs) > k) and type(kwargs[keys[k + 1]]) is not str:
                arg.doRawDataCorrect = bool(kwargs[keys[k + 1]])
                k += 2
            else:
                arg.doRawDataCorrect = True
                k += 1

        else:
            raise ValueError('Argument not recognized.')

    ###########################


    t0 = time()
    with open(filename,'rb') as f:

        # get file size
        f.seek(0,os.SEEK_END)
        fileSize = f.tell()

        # start of actual measurement data (sans header)
        f.seek(0,os.SEEK_SET)

        firstInt  = np.fromfile(f,dtype=np.uint32,count=1)[0]
        secondInt = np.fromfile(f,dtype=np.uint32,count=1)[0]

        # lazy software version check (VB or VD?)
        if (firstInt < 10000) and (secondInt <= 64):
            version = 'vd'
            logging.info('Software version: VD')

            # number of different scans in file stored in 2nd in
            NScans = secondInt
            measID = np.fromfile(f,dtype=np.uint32,count=1)[0]
            fileID = np.fromfile(f,dtype=np.uint32,count=1)[0]
            measOffset = np.zeros(NScans)
            measLength = np.zeros(NScans)

            for k in range(NScans):
                measOffset[k] = np.fromfile(f,dtype=np.uint64,count=1)[0]
                measLength[k] = np.fromfile(f,dtype=np.uint64,count=1)[0]
                f.seek(152 - 16,os.SEEK_CUR)

        else:
            # in VB versions, the first 4 bytes indicate the beginning of the
            # raw data part of the file
            version = 'vb'
            logging.info('Software version: VB')
            NScans = 1 # VB does not support multiple scans in one file
            measOffset = np.zeros(NScans)
            measLength = np.zeros(NScans)
            measOffset[0] = 0
            measLength[0] = fileSize


        # SRY read data correction factors
        #  do this for all VB datasets, so that the factors are available later
        #  in the image_obj if the user chooses to set the correction flag
        if version == 'vb': # not implemented/tested for vd, yet
            datStart = measOffset[0] + firstInt
            f.seek(0)

            rawfactors = None
            while (f.tell() < datStart) and rawfactors is None:
                line = f.readline().rstrip()

                # find the section of the protocol
                # note: the factors are also available in <ParamArray."CoilSelects">
                # along with element name and FFT scale, but the parsing is
                # significantly more difficult
                if b'<ParamArray."axRawDataCorrectionFactor">' in line:
                    while f.tell() < datStart:
                        line = f.readline().rstrip()

                        # find the line with correction factors
                        # the factors are on the first line that begins with this
                        # pattern
                        if b'{ {  { ' in line:
                            line = line.replace(b'}  { } ',b'0.0').replace(b'{',b'').replace(b'}',b'').decode()
                            rawfactors = np.loadtxt(StringIO(line),dtype=float)

                            # this does not work in this location because the MDHs
                            # have not been parsed yet
                            #                    if (length(rawfactors) ~= 2*max(image_obj.NCha))
                            #                       error('Number of raw factors (%f) does not equal channel count (%d)', length(rawfactors)/2, image_obj.NCha);
                            #                    end;

                            # We should have an even amount of rawfactors
                            if np.mod(rawfactors.size,2):
                                raise ValueError('Error reading rawfactors')

                            # note the transpose, this makes the vector
                            # multiplication during the read easier
                            arg['rawDataCorrectionFactors'] = rawfactors[::2] + 1j*rawfactors[1::2]
                            break

            logging.info('Done reading raw data correction factors')

        # data will be read in two steps (two while loops):
        #   1) reading all MDHs to find maximum line no., partition no.,... for
        #      ima, ref,... scan
        #   2) reading the data
        logging.info('Time ellapsed: %g sec' % (time() - t0))
        t0 = time()

        twix_obj = []

        # print reader version information
        logging.info('Reader version: %s' % twix_map_obj.readerVersion())

        for s in trange(NScans,desc='Read MDHS'):
            cPos = measOffset[s]
            f.seek(int(cPos),os.SEEK_SET)
            hdr_len = np.fromfile(f,dtype=np.uint32,count=1)[0]

            # read header and calculate regridding (optional)
            rstraj = None
            if arg['bReadHeader']:
                twix_obj.append({})
                twix_obj[s]['hdr'],rstraj = read_twix_hdr(f)

            # declare data objects:
            twix_obj[s]['image']         = twix_map_obj(arg,'image',filename,version,rstraj)
            twix_obj[s]['noise']         = twix_map_obj(arg,'noise',filename,version)
            twix_obj[s]['phasecor']      = twix_map_obj(arg,'phasecor',filename,version,rstraj)
            twix_obj[s]['phasestab']     = twix_map_obj(arg,'phasestab',filename,version,rstraj)
            twix_obj[s]['phasestabRef0'] = twix_map_obj(arg,'phasestab_ref0',filename,version,rstraj)
            twix_obj[s]['phasestabRef1'] = twix_map_obj(arg,'phasestab_ref1',filename,version,rstraj)
            twix_obj[s]['refscan']       = twix_map_obj(arg,'refscan',filename,version,rstraj)
            twix_obj[s]['refscanPC']     = twix_map_obj(arg,'refscan_phasecor',filename,version,rstraj)
            twix_obj[s]['refscanPS']     = twix_map_obj(arg,'refscan_phasestab',filename,version,rstraj)
            twix_obj[s]['refscanPSRef0'] = twix_map_obj(arg,'refscan_phasestab_ref0',filename,version,rstraj)
            twix_obj[s]['refscanPSRef1'] = twix_map_obj(arg,'refscan_phasestab_ref1',filename,version,rstraj)
            twix_obj[s]['RTfeedback']    = twix_map_obj(arg,'rtfeedback',filename,version,rstraj)
            twix_obj[s]['vop']           = twix_map_obj(arg,'vop',filename,version); # tx-array rf pulses

            # jump to first mdh
            cPos += hdr_len
            f.seek(int(cPos),os.SEEK_SET)

            # find all mdhs and save them in binary form, first:
            # logging.info('Scan %d/%d, read all mdhs:' % (s+1,NScans))

            mdh_blob,filePos,isEOF = loop_mdh_read(f,version,NScans,s,measOffset[s],measLength[s]) # uint8; size: [ byteMDH  Nmeas ]

            cPos = filePos[-1]
            filePos = filePos[:-1]

            # get mdhs and masks for each scan, no matter if noise, image, RTfeedback etc:
            mdh,mask = evalMDH(mdh_blob,version) # this is quasi-instant (< 1s) :-)
            del mdh_blob # earmark for garbage collection

            # Assign mdhs to their respective scans and parse it in the correct twix objects.
            if arg['bReadImaScan']:
                # tmpMdh = {}
                # for key in mdh.keys():
                #     tmpMdh[key] = mdh[key][mask['MDH_IMASCAN']]
                tmpMdh = { key: mdh[key][mask['MDH_IMASCAN']] for key in mdh.keys() }
                twix_obj[s]['image'].readMDH(tmpMdh,filePos[mask['MDH_IMASCAN']])

            if arg['bReadNoiseScan']:
                twix_obj[s]['noise'].readMDH({ key: mdh[key][mask['MDH_NOISEADJSCAN']] for key in mdh.keys() },filePos[mask['MDH_NOISEADJSCAN']])

            if arg['bReadRefScan']:
                isCurrScan = np.logical_and(
                    np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN']),
                    np.logical_or.reduce((
                        mask['MDH_PHASCOR'],
                        mask['MDH_PHASESTABSCAN'],
                        mask['MDH_REFPHASESTABSCAN'],
                        mask['MDH_RTFEEDBACK'],
                        mask['MDH_HPFEEDBACK']
                    ))
                )
                twix_obj[s]['refscan'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])

            if arg['bReadRTfeedback']:
                isCurrScan = np.logical_and(np.logical_or(mask['MDH_RTFEEDBACK'],mask['MDH_HPFEEDBACK']),~mask['MDH_VOP'])
                twix_obj[s]['RTfeedback'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])

                isCurrScan = np.logical_and(mask['MDH_RTFEEDBACK'],mask['MDH_VOP'])
                twix_obj[s]['vop'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])

            if arg['bReadPCScan']:
                # logic really correct?

                isCurrScan = np.logical_and(mask['MDH_PHASCOR'],~np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN']))
                twix_obj[s]['phasecor'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])

                isCurrScan = np.logical_and(mask['MDH_PHASCOR'],np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN']))
                twix_obj[s]['refscanPC'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])

            if arg['bReadPhaseStab']:
                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_PHASESTABSCAN'],~mask['MDH_REFPHASESTABSCAN']),
                    np.logical_or(~mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['phasestab'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_PHASESTABSCAN'],~mask['MDH_REFPHASESTABSCAN']),
                    np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['refscanPS'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_REFPHASESTABSCAN'],~mask['MDH_PHASESTABSCAN']),
                    np.logical_or(~mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['phasestabRef0'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_REFPHASESTABSCAN'],~mask['MDH_PHASESTABSCAN']),
                    np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['refscanPSRef0'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_REFPHASESTABSCAN'],mask['MDH_PHASESTABSCAN']),
                    np.logical_or(~mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['phasestabRef1'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


                isCurrScan = np.logical_and(
                    np.logical_and(mask['MDH_REFPHASESTABSCAN'],mask['MDH_PHASESTABSCAN']),
                    np.logical_or(mask['MDH_PATREFSCAN'],mask['MDH_PATREFANDIMASCAN'])
                )
                twix_obj[s]['refscanPSRef1'].readMDH({ key: mdh[key][isCurrScan] for key in mdh.keys() },filePos[isCurrScan])


            for scan in [ 'image', 'noise', 'phasecor', 'phasestab',
                         'phasestabRef0', 'phasestabRef1', 'refscan',
                         'refscanPC', 'refscanPS', 'refscanPSRef0',
                         'refscanPSRef1', 'RTfeedback', 'vop' ]:

                # remove unused fields
                if twix_obj[s][scan].NAcq == 0:
                    twix_obj[s].pop(scan)
                else:
                    if isEOF:
                        # recover from read error
                        twix_obj[s][scan].tryAndFixLastMdh()
                    else:
                        twix_obj[s][scan].clean()

        if NScans == 1:
            twix_obj = twix_obj[0]

        return(twix_obj)




def loop_mdh_read(f,version,Nscans,scan,measOffset,measLength):
    #  Goal of this function is to gather all mdhs in the dat file and store them
    #  in binary form, first. This enables us to evaluate and parse the stuff in
    #  a MATLAB-friendly (vectorized) way. We also yield a clear separation between
    #  a lengthy loop and other expressions that are evaluated very few times.
    #
    #  The main challenge is that we never know a priori, where the next mdh is
    #  and how many there are. So we have to actually evaluate some mdh fields to
    #  find the next one.
    #
    #  All slow things of the parsing step are found in the while loop.
    #  => It is the (only) place where micro-optimizations are worthwhile.
    #
    #  The current state is that we are close to sequential disk I/O times.
    #  More fancy improvements may be possible by using workers through parfeval()
    #  or threads using a java class (probably faster + no toolbox):
    #  http://undocumentedmatlab.com/blog/explicit-multi-threading-in-matlab-part1

    # raise NotImplementedError()

    if version == 'vb':
        isVD    = False
        byteMDH = 128
    elif version == 'vd':
        isVD    = True
        byteMDH = 184
        szScanHeader    = 192 # [bytes]
        szChannelHeader =  32 # [bytes]
    else:
        # arbitrary assumptions:
        isVD    = False
        byteMDH = 128
        logging.warning('Software version "%s" is not supported.' % version)

    cPos          = f.tell()
    n_acq         = 0
    allocSize     = 4096
    ulDMALength   = byteMDH
    isEOF         = False
    last_progress = 0

    mdh_blob = np.zeros((byteMDH,0),dtype=np.uint8)
    szBlob   = mdh_blob.shape[1]
    filePos  = np.zeros(0,dtype=type(cPos))  # avoid bug in Matlab 2013b: https://scivision.co/matlab-fseek-bug-with-uint64-offset/

    f.seek(int(cPos),os.SEEK_SET)

    # ======================================
    #   constants and conditional variables
    # ======================================
    bit_0 = np.uint8(2**0)
    bit_5 = np.uint8(2**5)
    mdhStart = 1 - byteMDH

    u8_000 = np.zeros((3,1),np.uint8) # for comparison with data_u8(1:3)


    # 20 fill bytes in VD (21:40)
    evIdx   = np.uint8(20 + 20*isVD) # 1st byte of evalInfoMask
    dmaIdx  = np.uint8(np.arange(29,32+1) + 20*isVD) - 1 # to correct DMA length using NCol and NCha
    if isVD:
        dmaOff  = szScanHeader
        dmaSkip = szChannelHeader
    else:
        dmaOff  = 0
        dmaSkip = byteMDH

    # ======================================

    with tqdm(total=measLength,unit_scale=True,leave=False,desc='Scan ID %d/%d' % (scan+1,Nscans)) as pbar:
        while True:
            #  Read mdh as binary (uint8) and evaluate as little as possible to know...
            #    ... where the next mdh is (ulDMALength / ushSamplesInScan & ushUsedChannels)
            #    ... whether it is only for sync (MDH_SYNCDATA)
            #    ... whether it is the last one (MDH_ACQEND)
            #  evalMDH() contains the correct and readable code for all mdh entries.

            try:
                #  read everything and cut out the mdh
                data_u8 = np.fromfile(f,count=int(ulDMALength),dtype=np.uint8)
                data_u8 = data_u8[mdhStart + data_u8.size - 1:]
            except:
                logging.error('An unexpected read error occurred at this byte offset: %d (%g GiB)' % (cPos,cPos/1024**3))
                logging.error('Will stop reading now.')
                isEOF = True
                break

            # the initial 8 bit from evalInfoMask are enough
            bitMask = data_u8[evIdx]

            if np.array_equal(data_u8[:3],u8_000) or np.bitwise_and(bitMask,bit_0): # probably ulDMALength == 0, # MDH_ACQEND
                # ok, look closer if really all *4* bytes are 0:
                data_u8[3] = np.unpackbits(data_u8[3].view(np.uint8))[0] # ubit24: keep only 1 bit from the 4th byte
                ulDMALength = (data_u8[:4].view(np.uint32)).astype(np.double)[0]

                if (ulDMALength == 0) or np.bitwise_and(bitMask,bit_0):
                    cPos += ulDMALength
                    # jump to next full 512 bytes
                    if np.mod(cPos,512):
                        cPos += 512 - np.mod(cPos,512)
                    break

            if np.bitwise_and(bitMask,bit_5): # MDH_SYNCDATA
                data_u8[3] = np.unpackbits(data_u8[3].view(np.uint8))[0] # ubit24: keep only 1 bit from the 4th byte
                ulDMALength = (data_u8[:4].view(np.uint32)).astype(np.double)[0]
                cPos += ulDMALength
                continue

            # pehses: the pack bit indicates that multiple ADC are packed into one
            # DMA, often in EPI scans (controlled by fRTSetReadoutPackaging in IDEA)
            # since this code assumes one adc (x NCha) per DMA, we have to correct
            # the "DMA length"
            #     if mdh.ulPackBit
            # it seems that the packbit is not always set correctly
            NCol_NCha = data_u8[dmaIdx].view(np.uint16).astype(np.double) # [ushSamplesInScan  ushUsedChannels]
            ulDMALength = dmaOff + (8*NCol_NCha[0] + dmaSkip)*NCol_NCha[1]

            # grow arrays in batches
            if (n_acq + 1) > szBlob:
                mdh_blob = np.concatenate((mdh_blob,np.zeros((mdh_blob.shape[0],allocSize),dtype=mdh_blob.dtype)),axis=-1)
                filePos = np.concatenate((filePos,np.zeros(allocSize)))
                szBlob = mdh_blob.shape[1]

            mdh_blob[:,n_acq] = data_u8
            filePos[n_acq] = cPos

            n_acq += 1
            pbar.update(cPos - measOffset - pbar.n)
            cPos += ulDMALength

    if isEOF:
        n_acq -= 1    # ignore the last attempt

    filePos = np.concatenate((filePos,np.array([cPos])),axis=-1)  # save pointer to the next scan

    # discard overallocation:
    mdh_blob = mdh_blob[:,:n_acq]
    filePos = filePos[:n_acq+1]

    # logging.info('%8.1f MB read in %4.0f s' % (measLength/1024**2,np.round(time() - t0)))
    return(mdh_blob,filePos,isEOF)



def evalMDH(mdh_blob,version):
    # see pkg/MrServers/MrMeasSrv/SeqIF/MDH/mdh.h
    # and pkg/MrServers/MrMeasSrv/SeqIF/MDH/MdhProxy.h

    if not np.issubdtype(mdh_blob.dtype,np.uint8):
        raise ValueError('mdh data must be a uint8 array, not a %s array!' % mdh_blob.dtype)

    if version[-1] == 'd':
        isVD = True
        mdh_blob = mdh_blob[list(range(20)) + list(range(40,mdh_blob.shape[0])),:] # remove 20 unnecessary bytes
    else:
        isVD = False

    Nmeas = mdh_blob.shape[1]

    mdh = {}
    mdh['ulPackBit'] = np.unpackbits(mdh_blob[3,:][:,None],axis=1)[:,1]
    # keep 6 relevant bits
    bits = np.unpackbits(mdh_blob[3,:][:,None],axis=1)
    bits[:,[ 6,7 ]] = 0
    mdh['ulPCI_rx'] = np.packbits(bits,axis=1).squeeze()
    # ubit24: keep only 1 bit from the 4th byte
    mdh_blob[3,:] = np.unpackbits(mdh_blob[3,:][:,None],axis=1)[:,0]

    data_uint32 = mdh_blob[:76,:].flatten('F').view(np.uint32)
    data_uint16 = mdh_blob[28:,:].flatten('F').view(np.uint16)
    data_single = mdh_blob[68:,:].flatten('F').view(np.single)

    data_uint32 = np.reshape(data_uint32,(Nmeas,-1))
    data_uint16 = np.reshape(data_uint16,(Nmeas,-1))
    data_single = np.reshape(data_single,(Nmeas,-1))

                                                             # byte pos.
    # %mdh.ulDMALength               = data_uint32(:,1);     #  1 :   4
    mdh['lMeasUID']                   = data_uint32[:,1]     #  5 :   8
    mdh['ulScanCounter']              = data_uint32[:,2]     #  9 :  12
    mdh['ulTimeStamp']                = data_uint32[:,3]     # 13 :  16
    mdh['ulPMUTimeStamp']             = data_uint32[:,4]     # 17 :  20
    mdh['aulEvalInfoMask']            = data_uint32[:,5:7]   # 21 :  28
    mdh['ushSamplesInScan']           = data_uint16[:,0]     # 29 :  30
    mdh['ushUsedChannels']            = data_uint16[:,1]     # 31 :  32
    mdh['sLC']                        = data_uint16[:,2:16]  # 33 :  60
    mdh['sCutOff']                    = data_uint16[:,16:18] # 61 :  64
    mdh['ushKSpaceCentreColumn']      = data_uint16[:,18]    # 66 :  66
    mdh['ushCoilSelect']              = data_uint16[:,19]    # 67 :  68
    mdh['fReadOutOffcentre']          = data_single[:, 0]    # 69 :  72
    mdh['ulTimeSinceLastRF']          = data_uint32[:,18]    # 73 :  76
    mdh['ushKSpaceCentreLineNo']      = data_uint16[:,24]    # 77 :  78
    mdh['ushKSpaceCentrePartitionNo'] = data_uint16[:,25]    # 79 :  80


    if isVD:
        mdh['SlicePos']               = data_single[:, 3:10] #  81 : 108
        mdh['aushIceProgramPara']     = data_uint16[:,40:64] # 109 : 156
        mdh['aushFreePara']           = data_uint16[:,64:68] # 157 : 164
    else:
        mdh['aushIceProgramPara']     = data_uint16[:,26:30] #  81 :  88
        mdh['aushFreePara']           = data_uint16[:,30:34] #  89 :  96
        mdh['SlicePos']               = data_single[:, 7:14] #  97 : 124



    # inlining of evalInfoMask
    mask = {}
    evalInfoMask1 = mdh['aulEvalInfoMask'][:,0]
    mask['MDH_ACQEND']            = np.bitwise_and(evalInfoMask1,2**0).astype(bool)
    mask['MDH_RTFEEDBACK']        = np.bitwise_and(evalInfoMask1,2**1).astype(bool)
    mask['MDH_HPFEEDBACK']        = np.bitwise_and(evalInfoMask1,2**2).astype(bool)
    mask['MDH_SYNCDATA']          = np.bitwise_and(evalInfoMask1,2**5).astype(bool)
    mask['MDH_RAWDATACORRECTION'] = np.bitwise_and(evalInfoMask1,2**10).astype(bool)
    mask['MDH_REFPHASESTABSCAN']  = np.bitwise_and(evalInfoMask1,2**14).astype(bool)
    mask['MDH_PHASESTABSCAN']     = np.bitwise_and(evalInfoMask1,2**15).astype(bool)
    mask['MDH_SIGNREV']           = np.bitwise_and(evalInfoMask1,2**17).astype(bool)
    mask['MDH_PHASCOR']           = np.bitwise_and(evalInfoMask1,2**21).astype(bool)
    mask['MDH_PATREFSCAN']        = np.bitwise_and(evalInfoMask1,2**22).astype(bool)
    mask['MDH_PATREFANDIMASCAN']  = np.bitwise_and(evalInfoMask1,2**23).astype(bool)
    mask['MDH_REFLECT']           = np.bitwise_and(evalInfoMask1,2**24).astype(bool)
    mask['MDH_NOISEADJSCAN']      = np.bitwise_and(evalInfoMask1,2**25).astype(bool)
    mask['MDH_VOP']               = np.bitwise_and(mdh['aulEvalInfoMask'].flatten('F')[1],2**(53-32)).astype(bool) # was 0 in VD
    mask['MDH_IMASCAN']           = np.ones(Nmeas,dtype=bool)


    noImaScan = np.logical_or.reduce((
        mask['MDH_ACQEND'],
        mask['MDH_RTFEEDBACK'],
        mask['MDH_HPFEEDBACK'],
        mask['MDH_PHASCOR'],
        mask['MDH_NOISEADJSCAN'],
        mask['MDH_PHASESTABSCAN'],
        mask['MDH_REFPHASESTABSCAN'],
        mask['MDH_SYNCDATA'],
        np.logical_and(mask['MDH_PATREFSCAN'],~mask['MDH_PATREFANDIMASCAN'])
    ))
    mask['MDH_IMASCAN'][noImaScan] = False

    return(mdh,mask)

if __name__  == '__main__':

    import matplotlib.pyplot as plt
    from mr_utils import view

    # mapVBVD()
    # mapVBVD('meas_MID00026_FID08491_t1_starvibe_60spokes_benz')
    #
    # mapVBVD(26)


    # val = mapVBVD(362)
    # print(val['image'].dataSize())

    # Follow the tutorial

    # twix = mapVBVD('meas_MID00026_FID08491_t1_starvibe_60spokes_benz.dat')
    # print('Num images: %d' % len(twix))
    #
    # twix = twix[1]
    # # for p in [ x for x in dir(twix['image']) if x[0] != '_' ]:
    # #     print(p,getattr(twix['image'],p))
    # print('dataSize',twix['image'].dataSize())
    # print('NAcq',twix['image'].NAcq)
    # print('getSqzDims',twix['image'].getSqzDims())
    # print('getSqzSize',twix['image'].getSqzSize())
    #
    # twix['image'].setFlagDoAverage(True)
    # twix['image'].setFlagRemoveOS(True)
    #
    # print('dataSize',twix['image'].dataSize())
    #
    # data = twix['image'].readData()
    # view(data)

    twix = mapVBVD(362)
    twix['image'].setFlagDoAverage(True)
    twix['image'].setFlagRemoveOS(True)
    data = twix['image'].readData()
    print(data.shape)

    test_im = data[:,0,:,...].squeeze()
    print(test_im.shape)
    view(test_im,log=True)
    # plt.imshow(np.abs(test_im[:,:]))
    # plt.show()
