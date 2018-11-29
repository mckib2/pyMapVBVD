import numpy as np
import logging
import os
from time import time

class twix_map_obj(object):
    '''Class to hold information about raw data from siemens MRI scanners.

    (currently VB and VD software versions are supported and tested).

    Author: Philipp Ehses (philipp.ehses@tuebingen.mpg.de), Aug/19/2011

    Modified by Wolf Blecher (wolf.blecher@tuebingen.mpg.de), Apr/26/2012
    Added reorder index to indicate which lines are reflected
    Added slice position for sorting, Mai/15/2012

    Order of many mdh parameters are now stored (including the reflected ADC
    bit); PE, Jun/14/2012

    data is now 'memory mapped' and not read until demanded;
    (see mapVBVD for a description) PE, Aug/02/2012

    twix_obj.image.unsorted now returns the data in its acq. order
    [NCol,NCha,nsamples in acq. order], all average flags don't have an
    influence on the output, but 'flagRemoveOS' still works, PE, Sep/04/13
    '''

    def __init__(self,arg=None,dataType=None,fname=None,version=None,rstraj=None):

        if dataType is None:
            self.dataType = 'image'
        else:
            self.dataType = dataType.lower()

        self.filename         = fname
        self.softwareVersion  = version

        self.IsReflected      = []
        self.IsRawDataCorrect = [] # %SRY
        self.NAcq             = 0
        self.isBrokenFile     = False

        self.dataDims = [ 'Col','Cha','Lin','Par','Sli','Ave','Phs', 'Eco','Rep','Set','Seg','Ida','Idb','Idc','Idd','Ide' ]

        self.arg = {}
        self.freadInfo = {}
        self.setDefaultFlags()


        if arg is not None:
            # % copy relevant arguments from mapVBVD argument list
            names = list(arg.keys())
            for k in range(len(names)):
                if names[k] in self.arg:
                    self.arg[names[k]] = arg[names[k]]

        self.flagAverageDim['Ave' in self.dataDims] = self.arg['doAverage']
        self.flagAverageDim['Rep' in self.dataDims] = self.arg['averageReps']
        self.flagAverageDim['Set' in self.dataDims] = self.arg['averageSets']
        self.flagAverageDim['Seg' in self.dataDims] = self.arg['ignoreSeg']

        if self.softwareVersion == 'vb':
            # % every channel has its own full mdh
            self.freadInfo['szScanHeader']    =   0 # % [bytes]
            self.freadInfo['szChannelHeader'] = 128 # % [bytes]
            self.freadInfo['iceParamSz']      =   4
        elif self.softwareVersion == 'vd':
            if (self.arg['doRawDataCorrect']):
                raise ValueError('raw data correction for VD not supported/tested yet')

            self.freadInfo['szScanHeader']    = 192 # % [bytes]
            self.freadInfo['szChannelHeader'] =  32 # % [bytes]
            self.freadInfo['iceParamSz']      =  24 # % vd version supports up to 24 ice params
        else:
            raise ValueError('software version not supported')


        if rstraj is not None:
            self.rampSampTrj = rstraj
        else:
            self.rampSampTrj        = []
            self.arg['rampSampRegrid'] = False

#     # % Copy function - replacement for matlab.mixin.Copyable.copy() to create object copies
#     # % from http://undocumentedmatlab.com/blog/general-use-object-copy
#     def copy(self):
#         # isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
#         # if isOctave || verLessThan('matlab','7.11')
#         #     % R2010a or earlier - serialize via temp file (slower)
#         #     fname = [tempname '.mat'];
#         #     save(fname, 'obj');
#         #     newObj = load(fname);
#         #     newObj = newObj.obj;
#         #     delete(fname);
#         # else
#         #     % R2010b or newer - directly in memory (faster)
#         #     objByteArray = getByteStreamFromArray(this);
#         #     newObj = getArrayFromByteStream(objByteArray);
#         raise NotImplementedError()
#
    def readMDH(self,mdh,filePos):
        '''Extract all values in all MDHs at once.

        data types:
        Use double for everything non-logical, both ints and floats. Seems the
        most robust way to avoid unexpected cast-issues with very nasty side effects.
        Examples: eps(single(16777216)) == 2
                  uint32( 10 ) - uint32( 20 ) == 0
                  uint16(100) + 1e5 == 65535
                  size(1 : 10000 * uint16(1000)) ==  [1  65535]

        The 1st example always hits the timestamps.
        '''

        if len(mdh) == 0:
            return

        self.NAcq     = filePos.size
        sLC           = mdh['sLC'].astype(np.double)
        evalInfoMask1 = mdh['aulEvalInfoMask'][:,0]

        # save mdh information for each line
        self.NCol       = mdh['ushSamplesInScan'].astype(np.double)
        self.NCha       = mdh['ushUsedChannels'].astype(np.double)
        self.Lin        = sLC[:,0]
        self.Ave        = sLC[:,1]
        self.Sli        = sLC[:,2]
        self.Par        = sLC[:,3]
        self.Eco        = sLC[:,4]
        self.Phs        = sLC[:,5]
        self.Rep        = sLC[:,6]
        self.Set        = sLC[:,7]
        self.Seg        = sLC[:,8]
        self.Ida        = sLC[:,9]
        self.Idb        = sLC[:,10]
        self.Idc        = sLC[:,11]
        self.Idd        = sLC[:,12]
        self.Ide        = sLC[:,13]

        self.centerCol   = mdh['ushKSpaceCentreColumn'].astype(np.double)
        self.centerLin   = mdh['ushKSpaceCentreLineNo'].astype(np.double)
        self.centerPar   = mdh['ushKSpaceCentrePartitionNo'].astype(np.double)
        self.cutOff      = mdh['sCutOff'].astype(np.double)
        self.coilSelect  = mdh['ushCoilSelect'].astype(np.double)
        self.ROoffcenter = mdh['fReadOutOffcentre'].astype(np.double)
        self.timeSinceRF = mdh['ulTimeSinceLastRF'].astype(np.double)
        self.IsReflected = np.bitwise_and(evalInfoMask1,2**24).astype(bool)
        self.scancounter = mdh['ulScanCounter'].astype(np.double)
        self.timestamp   = mdh['ulTimeStamp'].astype(np.double)
        self.pmutime     = mdh['ulPMUTimeStamp'].astype(np.double)
        self.IsRawDataCorrect = np.bitwise_and(evalInfoMask1,2**10).astype(bool)
        self.slicePos    = mdh['SlicePos']
        self.iceParam    = mdh['aushIceProgramPara']
        self.freeParam   = mdh['aushFreePara']

        self.memPos = filePos


    def tryAndFixLastMdh(self):
        # eofWarning = [mfilename() ]  # % We have it inside this.readData()
        eofWarning = '%s :UnxpctdEOF' % __file__
        logging.warning(eofWarning)

        isLastAcqGood = False
        cnt = 0

        while (not isLastAcqGood) and (self.NAcq > 0) and (cnt < 100):
            try:
                self.clean()
                self.unsorted(self.NAcq)
                isLastAcqGood = True
            except:
                self.isBrokenFile = True
                self.NAcq = self.NAcq - 1

            cnt += 1


    def clean(self):

        if self.NAcq == 0:
            return

        # Cut mdh data to actual size. Maybe we rejected acquisitions at the end
        # due to read errors.
        fields = [ 'NCol', 'NCha',
                   'Lin', 'Par', 'Sli', 'Ave', 'Phs', 'Eco', 'Rep',
                   'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide',
                   'centerCol'  ,   'centerLin',   'centerPar',           'cutOff',
                   'coilSelect' , 'ROoffcenter', 'timeSinceRF',      'IsReflected',
                   'scancounter',   'timestamp',     'pmutime', 'IsRawDataCorrect',
                   'slicePos'   ,    'iceParam',   'freeParam',           'memPos'  ]

        nack = self.NAcq
        idx = range(nack)

        for f in fields:
            if getattr(self,f).shape[-1] > nack: # rarely
                setattr(self,f,getattr(self,f)[:,idx]) # 1st dim: samples,  2nd dim acquisitions

        self.NLin = np.max(self.Lin) + 1
        self.NPar = np.max(self.Par) + 1
        self.NSli = np.max(self.Sli) + 1
        self.NAve = np.max(self.Ave) + 1
        self.NPhs = np.max(self.Phs) + 1
        self.NEco = np.max(self.Eco) + 1
        self.NRep = np.max(self.Rep) + 1
        self.NSet = np.max(self.Set) + 1
        self.NSeg = np.max(self.Seg) + 1
        self.NIda = np.max(self.Ida) + 1
        self.NIdb = np.max(self.Idb) + 1
        self.NIdc = np.max(self.Idc) + 1
        self.NIdd = np.max(self.Idd) + 1
        self.NIde = np.max(self.Ide) + 1

        # ok, let us assume for now that all NCol and NCha entries are
        # the same for all mdhs:
        self.NCol = self.NCol[0]
        self.NCha = self.NCha[0]

        if self.dataType == 'refscan':
            # pehses: check for lines with 'negative' line/partition numbers
            # this can happen when the reference scan line/partition range
            # exceeds the one of the actual imaging scan
            if self.NLin > 65500: #  %uint overflow check
                self.Lin  = np.mod(self.Lin + (65536 - np.min(self.Lin[self.Lin > 65500])),65536) + 1
                self.NLin = np.max(self.Lin)

            if self.NPar > 65500: # %uint overflow check
                self.Par  = np.mod(self.Par + (65536 - np.min(self.Par[self.Par > 65500])),65536) + 1
                self.NPar = np.max(self.Par)

        # to reduce the matrix sizes of non-image scans, the size
        # of the refscan_obj()-matrix is reduced to the area of the
        # actually scanned acs lines (the outer part of k-space
        # that is not scanned is not filled with zeros)
        # this behaviour is controlled by flagSkipToFirstLine which is
        # set to true by default for everything but image scans
        if not self.getFlagSkipToFirstLine():
            # the output matrix should include all leading zeros
            self.skipLin = 0
            self.skipPar = 0
        else:
            # otherwise, cut the matrix size to the start of the
            # first actually scanned line/partition (e.g. the acs/
            # phasecor data is only acquired in the k-space center)
            self.skipLin = np.min(self.Lin) - 1
            self.skipPar = np.min(self.Par) - 1

        NLinAlloc = np.max([ 1,self.NLin - self.skipLin ])
        NParAlloc = np.max([ 1,self.NPar - self.skipPar ])

        self.fullSize = [ self.NCol, self.NCha, NLinAlloc, NParAlloc,
                          self.NSli, self.NAve, self.NPhs, self.NEco,
                          self.NRep, self.NSet, self.NSeg, self.NIda,
                          self.NIdb, self.NIdc, self.NIdd, self.NIde ]

        nByte = self.NCha*(self.freadInfo['szChannelHeader'] + 8*self.NCol)

        # size for fread
        self.freadInfo['sz']    = np.array([2, nByte/8])
        # reshape size
        self.freadInfo['shape'] = np.array([self.NCol + self.freadInfo['szChannelHeader']/8, self.NCha])
        # we need to cut MDHs from fread data
        self.freadInfo['cut']   = self.freadInfo['szChannelHeader']/8 + np.array(list(range(int(self.NCol))))
#
#
#     def subsref(self,S):
#         # # % this is where the magic happens
#         # # % Now seriously. Overloading of the subsref-method and working
#         # # % with a gazillion indices got really messy really fast. At
#         # # % some point, I should probably clean this code up a bit. But
#         # # % good news everyone: It seems to work.
#         # if S[0].type == '.':
#         #     # % We don't want to manage method/variable calls, so we'll
#         #     # % simply call the built-in subsref-function in this case.
#         #     if nargout == 0:
#         #         varargout{1} = builtin('subsref', this, S); % CTR fix.
#         #     else
#         #         varargout      = cell(1, nargout);
#         #         [varargout{:}] = builtin('subsref', this, S);
#         #
#         #     return
#         #
#         # else:
#         #     raise ValueError('operator not supported')
#         #
#         #
#         # [selRange,selRangeSz,outSize] = this.calcRange(S(1));
#         #
#     	# # % calculate page table (virtual to physical addresses)
#         # # % this is now done every time, i.e. result is no longer saved in
#         # # % a property - slower but safer (and easier to keep track of updates)
#         # ixToRaw = this.calcIndices;
#         #
#         # tmp = reshape(1:prod(double(this.fullSize(3:end))), this.fullSize(3:end));
#         # tmp = tmp(selRange{3:end});
#         # ixToRaw = ixToRaw(tmp); clear tmp;
#         # ixToRaw = ixToRaw(:);
#         # # % delete all entries that point to zero (the "NULL"-pointer)
#         # notAcquired = (ixToRaw == 0);
#         # ixToRaw (notAcquired) = []; clear notAcquired;
#         #
#         # # % calculate ixToTarg for possibly smaller, shifted + segmented
#         # # % target matrix:
#         # cIx = ones(14, numel(ixToRaw), 'single');
#         # if ~this.flagAverageDim(3)
#         #     cIx( 1,:) = this.Lin(ixToRaw) - this.skipLin;
#         # end
#         # if ~this.flagAverageDim(4)
#         #     cIx( 2,:) = this.Par(ixToRaw) - this.skipPar;
#         # end
#         # if ~this.flagAverageDim(5)
#         #     cIx( 3,:) = this.Sli(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(6)
#         #     cIx( 4,:) = this.Ave(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(7)
#         #     cIx( 5,:) = this.Phs(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(8)
#         #     cIx( 6,:) = this.Eco(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(9)
#         #     cIx( 7,:) = this.Rep(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(10)
#         #     cIx( 8,:) = this.Set(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(11)
#         #     cIx( 9,:) = this.Seg(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(12)
#         #     cIx(10,:) = this.Ida(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(13)
#         #     cIx(11,:) = this.Idb(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(14)
#         #     cIx(12,:) = this.Idc(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(15)
#         #     cIx(13,:) = this.Idd(ixToRaw);
#         # end
#         # if ~this.flagAverageDim(16)
#         #     cIx(14,:) = this.Ide(ixToRaw);
#         # end
#         #
#         # # % make sure that indices fit inside selection range
#         # for k=3:numel(selRange)
#         #     tmp = cIx(k-2,:);
#         #     for l=1:numel(selRange{k})
#         #         cIx(k-2,tmp==selRange{k}(l)) = l;
#         #     end
#         # end
#         #
#         # sz = selRangeSz(3:end); % extra variable needed for octave compatibility
#         # ixToTarg = this.sub2ind_double(sz, cIx(1,:),cIx(2,:),cIx(3,:),...
#         #     cIx(4,:),cIx(5,:),cIx(6,:),cIx(7,:),cIx(8,:),cIx(9,:),...
#         #     cIx(10,:),cIx(11,:),cIx(12,:),cIx(13,:),cIx(14,:));
#         #
#         # mem = this.memPos(ixToRaw);
#         # # % sort mem for quicker access, sort cIxToTarg/Raw accordingly
#         # [mem,ix]  = sort(mem);
#         # ixToTarg = ixToTarg(ix);
#         # ixToRaw  = ixToRaw(ix);
#         # clear ix;
#         #
#         # # % For a call of type data{:,:,1:3} matlab expects more than one
#         # # % output variable (three in this case) and will throw an error
#         # # % otherwise. This is a lazy way (and the only one I know of) to
#         # # % fix this.
#         # varargout    = cell(1, nargout);
#         # varargout{1} = this.readData(mem,ixToTarg,ixToRaw,selRange,selRangeSz,outSize);
#         raise NotImplementedError()
#
    def unsorted(self,ival=None):
        # returns the unsorted data [NCol,NCha,#samples in acq. order]
        if ival is None:
            mem = self.memPos
        else:
            mem = self.memPos[ival]

        out = self.readData(mem)
        return(out)


    # def readData(self,cIxToTarg=None,cIxToRaw=None,selRange=None,selRangeSz=None,mem=None,outSize=None):
    def readData(self,
        col=None,
        cha=None,
        lin=None,
        par=None,
        sli=None,
        ave=None,
        phs=None,
        eco=None,
        rep=None,
        set=None,
        seg=None,

        mem=None
    ):

        if mem is None:
            mem = self.memPos

        selRange,selRangeSz,outSize = self.calcRange(col,cha,lin,par,sli,ave,phs,eco,rep,set,seg)

        cIx = np.zeros((14,mem.size),dtype=np.single)
        # Skip Col,Cha => start at index 2
        for ii in range(2,self.flagAverageDim.size):
            if not self.flagAverageDim[ii]:
                cIx[ii-2,:] = getattr(self,self.dataDims[ii])

                # Take care of skipLin and skipPar:
                if hasattr(self,'skip%s' % self.dataDims[ii]):
                    cIx[ii-2,:] -= getattr(self,'skip%s' % self.dataDims[ii])
        selR = np.array(selRangeSz[2:])

        # Flip around the first two for some reason...
        cIx[[0,1],:] = cIx[[1,0],:]
        selR[[0,1]] = selR[[1,0]]

        ixToTarg = np.ravel_multi_index([ x.astype(int) for x in np.split(cIx,cIx.shape[0],axis=0) ],selR,order='F')[0]
        ixToRaw = np.zeros(np.prod(self.fullSize[2:]).astype(int))
        ixToRaw[ixToTarg] = np.array(range(ixToTarg.size))
        ixToRaw = ixToRaw.astype(int)

        # For some reason it's different now for some reason
        # ixToTarg = np.ravel_multi_index([ x.astype(int) for x in np.split(cIx,cIx.shape[0],axis=0) ],selRangeSz[2:],order='C')[0]

        out = np.zeros(outSize,dtype='complex')
        out = np.reshape(out,(int(selRangeSz[0]),int(selRangeSz[1]),-1))

        if mem is None:
            out = np.reshape(out,outSize)
            return

        # cIxToTarg = self.cast2MinimalUint( cIxToTarg )

        # subsref overloading makes this.that-calls slow, so we need to
        # avoid them whenever possible
        szScanHeader = self.freadInfo['szScanHeader']
        readSize     = self.freadInfo['sz']
        readShape    = self.freadInfo['shape']
        readCut      = self.freadInfo['cut'].astype(int)
        keepOS       = np.concatenate((np.arange(0,self.NCol/4),np.arange(self.NCol*3/4,self.NCol))).astype(int)
        bRemoveOS    = self.arg['removeOS']
        bIsReflected = self.IsReflected[ixToRaw]
        bRegrid      = self.getFlagRampSampRegrid() and len(self.rampSampTrj)
        slicedata    = self.slicePos[ixToRaw,:]
        # SRY store information about raw data correction
        bDoRawDataCorrect = self.arg['doRawDataCorrect']
        bIsRawDataCorrect = self.IsRawDataCorrect[ixToRaw]
        isBrokenRead      = False
        if bDoRawDataCorrect:
            rawDataCorrect = self.arg['rawDataCorrectionFactors']

        print(readCut)

        # MiVö: Raw data are read line-by-line in portions of 2xNColxNCha float32 points (2 for complex).
        # Computing and sorting(!) on these small portions is quite expensive, esp. when
        # it employs non-sequential memory paths. Examples are non-linear k-space acquisition
        # or reflected lines.
        # This can be sped up if slightly larger blocks of raw data are collected, first.
        # Whenever a block is full, we do all those operations and save it in the final "out" array.
        # What's a good block size? Depends on data size and machine (probably L2/L3/L4 cache sizes).
        # So...? Start with a small block, measure the time-per-line and double block size until
        # a minimum is found. Seems sufficiently robust to end up in a close-to-optimal size for every
        # machine and data.
        blockSz   = 2          # size of blocks; must be 2^n; will be increased
        doLockblockSz = False  # whether blockSZ should be left untouched
        tprev     = np.inf        # previous time-per-line
        blockCtr  = 0
        blockInit = np.ones((readShape[0].astype(int),readShape[1].astype(int),blockSz),dtype=np.single)*-np.inf # init with garbage
        blockInit = blockInit.astype('complex')
        block     = blockInit.copy()

        if bRegrid:
            raise NotImplementedError()
            # v1       = single(1:selRangeSz(2))
            # v2       = single(1:blockSz)
            # rsTrj    = {self.rampSampTrj,v1,v2}
            # trgTrj   = linspace(min(self.rampSampTrj),max(self.rampSampTrj),self.NCol)
            # trgTrj   = {trgTrj,v1,v2}

        # counter for proper scaling of averages/segments
        count_ave = np.zeros(mem.size,dtype=np.single)
        kMax      = len(mem)-1  # max loop index

        with open(self.filename,'rb') as f:

            for k in range(kMax):
                # skip scan header
                f.seek(int(mem[k] + szScanHeader),os.SEEK_SET)

                # MiVö: With incomplete files fread() returns less than readSize points. The subsequent reshape will therefore error out.
                #       We could check if numel(raw) == prod(readSize), but people recommend exception handling for performance
                #       reasons. Do it.
                try:
                    raw = np.zeros(readSize.astype(int),dtype=np.single)
                    for ii in range(int(readSize[0])):
                        raw[ii,:] = np.fromfile(f,count=int(readSize[1]),dtype=np.single)

                    raw = np.reshape((raw[0,:] + 1j*raw[1,:]),readShape.astype(int))
                except:
                    logging.warning('WHOOPS')
                    offset_bytes = mem[k] + szScanHeader
                    remainingSz = readSize[1] - raw.shape[0]

                    # warning( [mfilename() ':UnxpctdEOF'],  ...
                    #           [ '\nAn unexpected read error occurred at this byte offset: %d (%g GiB)\n'...
                    #             'Actual read size is [%s], desired size was: [%s]\n'                    ...
                    #             'Will ignore this line and stop reading.\n'                             ...
                    #             '=== MATLABs error message ================\n'                          ...
                    #             exc.message                                                             ...
                    #             '\n=== end of error =========================\n'                        ...
                    #             ], offset_bytes, offset_bytes/1024**3, num2str(size(raw)), num2str(readSize.') )

                    # Reject this data fragment. To do so, init with the values of blockInit
                    # clear raw
                    print(readShape)
                    raw = np.zeros(readShape.astype(int))
                    raw[:int(np.prod(readShape))] = blockInit[0]
                    raw = np.reshape(raw,readShape.astype(int))
                    isBrokenRead = True  # remember it and bail out later

                block[:,:,blockCtr] = raw # fast serial storage in a cache array
                blockCtr += 1

                # Do expensive computations and reorderings on the gathered block.
                # Unfortunately, a lot of code is necessary, but that is executed much less
                # frequent, so its worthwhile for speed.
                # TODO: Do *everything* block-by-block
                if (blockCtr == blockSz) or (k == kMax) or (isBrokenRead and blockCtr > 1):
                    # s = tic   # % measure the time to process a block of data
                    s = time()

                    # remove MDH data from block:
                    block = block[readCut,:,:]

                    if bRegrid:
                        raise NotImplementedError()
                        # # correct for readout shifts
                        # # the nco frequency is always scaled to the max.
                        # # gradient amp and does account for ramp-sampling
                        # ro_shift = self.calcOffcenterShiftRO(slicedata(:,k))
                        # deltak = max(abs(diff(rsTrj{1})))
                        # phase = (0:self.NCol-1).T * deltak * ro_shift
                        # phase_factor = exp(1j*2*pi*(phase - ro_shift * rsTrj{1}))
                        # block = bsxfun(@times, block, phase_factor)
                        #
                        # # grid the data
                        # F = griddedInterpolant(rsTrj, block)
                        # block = F(trgTrj)


                    ix = np.array(range(1 + k - blockCtr,k+1)).astype(int)
                    if blockCtr != blockSz:
                        block = block[:,:,:blockCtr]


                    if np.isnan(block).any():
                        # keyboard
                        raise NotImplementedError()


                    if bRemoveOS: # remove oversampling in read

                        # a = 2
                        # block = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(block,axes=a),axis=a),axes=a)
                        # block *= np.sqrt(np.prod(np.take(block.shape,a)))
                        # block = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(block[keepOS,...],axes=a),axis=a),axes=a)
                        # block /= np.sqrt(np.prod(np.take(block.shape,a)))
                        #
                        block = np.fft.ifft(block,axis=0)
                        block = np.fft.fft(block[keepOS,:,:],axis=0)

                    if bDoRawDataCorrect and bIsRawDataCorrect[k]:
                        # SRY apply raw data correction if necessary
                        # block = bsxfun(@times, block, rawDataCorrect)
                        raise NotImplementedError()

                    isRefl = bIsReflected[ix]
                    block[:,:,isRefl] = np.flip(block[:,:,isRefl],axis=0)

                    # if ~isequal(selRange{1},':') or ~isequal(selRange{2},':'):
                    #     block = block[selRange{1}, selRange{2}, :] # a bit slow

                    sortIdx = np.sort(ixToTarg[ix])
                    I = np.argsort(ixToTarg[ix])
                    block = block[:,:,I]    # reorder according to sorted target indices

                    # Mark duplicate indices with 1; we'll have to treat them special for proper averaging
                    # Bonus: The very first storage can be made much faster, because it's in-place.
                    #        Matlab urgently needs a "+=" operater, which makes "A(:,:,idx) = A(:,:,idx) + B"
                    #        in-place and more readable.
                    isDupe = np.concatenate((np.array([ False ]),np.diff(sortIdx) == 0))

                    idx1 = sortIdx[~isDupe]    # acquired once in this block
                    idxN = sortIdx[ isDupe]    # acquired multiple times

                    count_ave[idx1] += 1

                    if idxN.size == 0:
                        # no duplicates
                        if np.all(count_ave[idx1] == 1): # first acquisition of this line
                            out[:,:,idx1] = block                              # fast
                        else:
                            out[:,:,idx1] += block              # slow

                    else:
                        out[:,:,idx1] += block[:,:,~isDupe]    # slower

                        block = block[:,:,isDupe]
                        for n in range(len(idxN)):
                            out[:,:,idxN[n]] += block[:,:,n] # snail :-)
                            count_ave[idxN[n]] += 1


                    # At the first few iterations, evaluate the spent time-per-line and decide
                    # what to do with the block size.
                    if not doLockblockSz:
                        t = 1e6 * (time() - s)/blockSz #  micro seconds

                        if t <= 1.1*tprev: # allow 10% inaccuracy. Usually bigger == better
                            # New block size was faster. Go a step further.
                            logging.info('Making bigger block...')
                            blockSz *= 2
                            blockInit = np.concatenate((blockInit,blockInit),axis=2)
                        else:
                            # ? regression; reset size and lock it
                            logging.info('Locking block size...')
                            blockSz = np.max([ blockSz/2,1 ])
                            blockInit = blockInit[:,:,:int(blockSz)]
                            doLockblockSz = True

                        if bRegrid:
                            raise NotImplementedError()
                            # rsTrj{3}  = single(1:blockSz)
                            # trgTrj{3} = rsTrj{3}

                        tprev = t


                    blockCtr = 0
                    block = blockInit.copy() # reset to garbage

                if isBrokenRead:
                    self.isBrokenFile = True
                    break


        # proper scaling (we don't want to sum our data but average it)
        # For large "out" bsxfun(@rdivide,out,count_ave) is incredibly faster than
        # bsxfun(@times,out,count_ave)!
        # @rdivide is also running in parallel, while @times is not. :-/
        if np.any(count_ave > 1):
            logging.info('Averaging...')
            # clearvars -except  out  count_ave  outSize
            print(count_ave)
            count_ave = np.maximum(np.ones(count_ave.shape),count_ave)
            out = np.apply_over_axes(lambda x,d: x/count_ave[d],out,range(out.ndim))

        out = np.reshape(out,outSize)
        return(out)


    def setDefaultFlags(self):
        # method to set flags to default values
        self.arg['removeOS']            = False
        self.arg['rampSampRegrid']      = False
        self.arg['doAverage']           = False
        self.arg['averageReps']         = False
        self.arg['averageSets']         = False
        self.arg['ignoreSeg']           = False
        self.arg['doRawDataCorrect']    = False
        self.flagAverageDim          = np.zeros(16,dtype=bool)

        if self.dataType in [ 'image','phasecor','phasestab' ]:
            self.arg['skipToFirstLine'] = False
        else:
            self.arg['skipToFirstLine'] = True

        if 'rawDataCorrectionFactors' not in self.arg:
            self.arg['rawDataCorrectionFactors'] = []

    def resetFlags(self):
        # method to reset flags to default values
        self.flagRemoveOS            = False
        self.flagRampSampRegrid      = False
        self.flagDoRawDataCorrect    = False
        self.flagAverageDim          = [False]*16

        if self.dataType in [ 'image','phasecor','phasestab' ]:
            self.arg['skipToFirstLine'] = False
        else:
            self.arg['skipToFirstLine'] = True


    @staticmethod
    def readerVersion():
        # returns utc-unixtime of last commit (from file precommit-unixtime)
        # return('Not Implemented')
        # p = fileparts(mfilename('fullpath'))
        # fid = fopen(fullfile(p, 'precommit_unixtime'))
        # versiontime = uint64(str2double(fgetl(fid)))
        # fclose(fid)
        return('VERSION GOES HERE')

    def dataSize(self):
        out = np.array(self.fullSize,dtype=int)

        if self.arg['removeOS']:
            # I don't think this is right...
            idx = [ ii for ii,x in enumerate(self.dataDims) if x == 'Col' ]
            out[idx] = self.NCol/2

        if self.flagAverageDim[0] or self.flagAverageDim[1]:
            logging.warn('averaging in col and cha dim not supported, resetting flag')
            self.flagAverageDim[0:2] = False

        out[self.flagAverageDim] = 1
        return(out)


    def getSqzDims(self):
        idx = np.where(self.dataSize() > 1)[0]
        out = [ self.dataDims[ii] for ii in idx ]
        return(out)


    def getSqzSize(self):
        idx = np.where(self.dataSize() > 1)[0]
        out = [ self.dataSize()[ii] for ii in idx ]
        return(out)


    def setFlagRemoveOS(self,val):
        # set method for removeOS
        self.arg['removeOS'] = bool(val)

    def getFlagRemoveOS(self):
        out = self.arg['removeOS']
        return(out)


    def setFlagDoAverage(self,val):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Ave' ]
        self.flagAverageDim[ix] = bool(val)


    def getFlagDoAverage(self):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Ave' ]
        out = self.flagAverageDim[ix]
        return(out)


    def setFlagAverageReps(self,val):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Rep' ]
        self.flagAverageDim[ix] = bool(val)


    def getFlagAverageReps(self):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Rep' ]
        out = self.flagAverageDim[ix]
        return(out)


    def setFlagAverageSets(self,val):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Set' ]
        self.flagAverageDim[ix] = bool(val)


    def getFlagAverageSets(self):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Set' ]
        out = self.flagAverageDim[ix]
        return(out)

    def setFlagIgnoreSeg(self,val):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Seg' ]
        self.flagAverageDim[ix] = bool(val)


    def getFlagIgnoreSeg(self):
        ix = [ ii for ii,x in enumerate(self.dataDims) if x == 'Seg' ]
        out = self.flagAverageDim[ix]
        return(out)
#
#     def set.flagSkipToFirstLine(self,val):
#         val = bool(val)
#         if val != self.arg.skipToFirstLine:
#             self.arg.skipToFirstLine = val
#
#             if self.arg.skipToFirstLine:
#                 self.skipLin = min(self.Lin) - 1
#                 self.skipPar = min(self.Par) - 1
#             else:
#                 self.skipLin = 0
#                 self.skipPar = 0
#
#             NLinAlloc = max(1, self.NLin - self.skipLin)
#             NParAlloc = max(1, self.NPar - self.skipPar)
#             self.fullSize(3:4) = [NLinAlloc NParAlloc]
#

    def getFlagSkipToFirstLine(self):
        out = self.arg['skipToFirstLine']
        return(out)

    def getFlagRampSampRegrid(self):
        out = self.arg['rampSampRegrid']
        return(out)

    def setFlagRampSampRegrid(self,val):
        val = bool(val)
        if (val and self.rampSampTrj is None):
            raise ValueError('No trajectory for regridding available')
        self.arg['rampSampRegrid'] = val
#
#     # %SRY: accessor methods for raw data correction
#     def get.flagDoRawDataCorrect(self):
#         out = self.arg.doRawDataCorrect
#         return(out)
#
#     def set.flagDoRawDataCorrect(self,val):
#         val = bool(val)
#         if (val and self.softwareVersion == 'vd':
#             raise ValueError('raw data correction for VD not supported/tested yet')
#         self.arg.doRawDataCorrect = val
#
#     def get.RawDataCorrectionFactors(self):
#         out = self.arg.rawDataCorrectionFactors
#         return(out)
#
#
#     def set.RawDataCorrectionFactors(self,val):
#         # %this may not work if trying to set the factors before NCha has
#         # %a meaningful value (ie before calling clean)
#         if (~isrow(val) or length(val) != self.NCha):
#             raise ValueError('RawDataCorrectionFactors must be a 1xNCha row vector')
#         self.arg.rawDataCorrectionFactors = val;
#
# # methods (Access='protected')
#     # % helper functions
#
#     def fileopen(self):
#         # % look out for unlikely event that someone is switching between
#         # % windows and unix systems:
#         [path,name,ext] = fileparts(self.filename)
#         self.filename   = fullfile(path,[name ext])
#
#         # % test access
#         if len(dir(self.filename)) == 0:
#             # % update path when file of same name can be found in current
#             # % working dir. -- otherwise throw error
#             [oldpath,name,ext] = fileparts(self.filename);
#             newloc = fullfile(pwd,[name ext]);
#             if len(dir(newloc)) == 1:
#                 print('Warning: File location updated from "%s" to current working directory.' % oldpath)
#                 self.filename = newloc
#             else:
#                 raise ValueError('File %s not found.' % self.filename)
#
#         fid = fopen(self.filename)
#         return(fid)
#
#
    # def calcRange(self,S):
    def calcRange(self,col,cha,lin,par,sli,ave,phs,eco,rep,set,seg):
        args = locals()
        args.pop('self')

        bSqueeze = False
#
#         switch S.type
#             case '()'
#                 bSqueeze = False
#             case '{}'
#                 bSqueeze = True
#
        dataSize = self.dataSize()
        selRange = {} #np.ones(len(dataSize)) #num2cell(ones(1,numel(self.dataSize)));

        outSize  = np.ones(len(dataSize))

        # if ( isempty(S.subs) || strcmpi(S.subs(1),'') ):
        if all(x is None for x in args.values()):
            # obj(): shortcut to select all data
            # unfortunately, matlab does not allow the statement
            # obj{}, so we can't use it...
            # alternative: obj{''} (obj('') also works)
            selRange = { ii:list(range(val)) for ii,val in enumerate(dataSize)  }
            # for k in range(len(dataSize)):
            #     selRange[k] = range(dataSize[k])

            if not bSqueeze:
                outSize = dataSize
            else:
                outSize = self.getSqzSize()

        else:
            sqzDims = self.getSqzDims()
            for k in [ ii for ii,x in enumerate(args.values()) if x is not None ]:
                # if not bSqueeze:
                #     cDim = k # nothing to do
                # else:
                #     # we need to rearrange selRange from squeezed
                #     # to original order
                #     # cDim = self.dataDims.index(sqzDims[k])
                #     cDim = k
                #     # print(cDim)
                cDim = k
                selRange[cDim] = args[list(args.keys())[k]]
                if type(selRange[cDim]) in [ int,float ]:
                    outSize[k] = selRange[cDim]
                    selRange[cDim] = [ selRange[cDim] ]
                elif type(selRange) in [ list ]:
                    outSize[k] = len(selRange[cDim])
                else:
                    raise ValueError('unknown indices')


            for ii,val in enumerate(selRange):
                if np.max(val) > dataSize[ii]:
                    raise ValueError('selection out of range')

        selRangeSz = [ len(val) for val in selRange.values() ]

        # now select all indices for the dims that are averaged
        fullSize = self.fullSize
        for k in [ ii for ii,val in enumerate(self.flagAverageDim) if val ]:
            selRange[k] = list(range(int(fullSize[k])))

        return(selRange,selRangeSz,outSize)


    def calcIndices(self):
        # calculate indices to target & source(raw)
        LinIx     = self.Lin - self.skipLin
        ParIx     = self.Par - self.skipPar
        sz = self.fullSize[2:] # extra variable needed for octave compatibility
        ixToTarget = self.sub2ind_double(sz,
            LinIx, ParIx, self.Sli, self.Ave, self.Phs, self.Eco,
            self.Rep, self.Set, self.Seg, self.Ida, self.Idb,
            self.Idc, self.Idd, self.Ide)

        # now calc. inverse index (page table: virtual to physical addresses)
        # indices of lines that are not measured are zero
        ixToRaw = np.zeros(np.prod(self.fullSize[2:]),dtype=np.double)

        ixToRaw[ixToTarget] = list(range(len(ixToTarget)))

        return(ixToRaw,ixToTarget)
#
# # methods (Static)
#     # % helper functions, accessible from outside via <classname>.function()
#     # % without an instance of the class.
#     @staticmethod
#     def calcOffcenterShiftRO(slicedata):
#         # % calculate ro offcenter shift from mdh's slicedata
#
#         # % slice position
#         pos = slicedata(1:3);
#
#         # %quaternion
#         a = slicedata(5);
#         b = slicedata(6);
#         c = slicedata(7);
#         d = slicedata(4);
#
#         read_dir = zeros(3,1);
#         read_dir(1) = 2 * (a * b - c * d);
#         read_dir(2) = 1 - 2 * (a^2 + c^2);
#         read_dir(3) = 2 * (b * c + a * d);
#
#         ro_shift = dot(pos, read_dir)
#         return(ro_shift)
#
    # @staticmethod
    # def sub2ind_double(self,sz,varargin):
    #     # %SUB2IND_double Linear index from multiple subscripts.
    #     # %   Works like sub2ind but always returns double
    #     # %   also slightly faster, but no checks
    #     # %========================================
    #     sz  = double(sz)
    #     ndx = double(varargin{end}) - 1
    #     for i in range(len(sz)-1:-1:1)
    #         ix  = double(varargin{i})
    #         ndx = sz[i]*ndx + ix - 1
    #     ndx += 1
    #     return(ndx)
#
#     @staticmethod
#     defcast2MinimalUint( N ):
#         Nmax = max( reshape(N,[],1) );
#         Nmin = min( reshape(N,[],1) );
#         if Nmin < 0 || Nmax > intmax('uint64')
#             return
#
#         if Nmax > intmax('uint32')
#             idxClass = 'uint64';
#         elseif Nmax > intmax('uint16')
#             idxClass = 'uint32';
#         else
#             idxClass = 'uint16';
#
#         N = cast( N, idxClass )
#         return(N)
