function export_mrst_schedule_control(deckfile, outfile, mrstroot)
%Dump MRST's SCHEDULE.control / SCHEDULE.step for a deck, as plain text.
%
% SYNOPSIS:
%   export_mrst_schedule_control(deckfile, outfile)
%   export_mrst_schedule_control(deckfile, outfile, mrstroot)
%
% DESCRIPTION:
%   readSCHEDULE splits the SCHEDULE section into one control per stretch
%   of well keywords between two DATES/TSTEP records.  PRSTCore ports that
%   splitting in PRSTCore/deckformat/deckinput/schedule_control.py; this
%   writes the MATLAB side of the comparison.
%
%   The format is deliberately line-oriented rather than a .mat file: a
%   control table is a cell array of mixed strings and doubles, which does
%   not survive loadmat in a shape worth asserting on.  Every row is
%   written as
%
%      row <control> <keyword> <rownumber> | <item> <item> ...
%
%   with numbers as %.10g, strings verbatim, and a defaulted numeric item
%   (MATLAB's empty matrix) as <empty>.
%
% PARAMETERS:
%   deckfile - .DATA file to read.
%   outfile  - text file to write.
%   mrstroot - MRST checkout to run.  Defaults to the MRST-0 tree, the
%              industrialised version PRSTCore's history matching follows.

   if nargin < 3 || isempty(mrstroot)
      mrstroot = fullfile(getenv('USERPROFILE'), 'Desktop', 'github', ...
                          'MRST-0', 'MRST');
   end
   run(fullfile(mrstroot, 'startup.m'));
   mrstModule add deckformat

   deck = readEclipseDeck(deckfile);
   S    = deck.SCHEDULE;

   fid = fopen(outfile, 'wt');
   if fid < 0
      error('Failed to open ''%s'' for writing.', outfile);
   end
   cleanup = onCleanup(@() fclose(fid));

   fprintf(fid, 'ncontrol %d\n', numel(S.control));
   fprintf(fid, 'nstep %d\n', numel(S.step.val));
   fprintf(fid, 'stepval');
   fprintf(fid, ' %.10g', S.step.val);
   fprintf(fid, '\n');
   fprintf(fid, 'stepctrl');
   fprintf(fid, ' %d', S.step.control);
   fprintf(fid, '\n');

   kws = {'WELSPECS', 'COMPDAT', 'WCONHIST', 'WCONINJH', ...
          'WCONPROD', 'WCONINJE', 'WCONINJ', 'WELTARG', 'GRUPTREE'};

   for c = 1 : numel(S.control)
      for k = 1 : numel(kws)
         kw = kws{k};
         if ~isfield(S.control(c), kw), continue; end
         t = S.control(c).(kw);
         fprintf(fid, 'nrow %d %s %d\n', c, kw, size(t, 1));
         for r = 1 : size(t, 1)
            fprintf(fid, 'row %d %s %d |', c, kw, r);
            for j = 1 : size(t, 2)
               v = t{r, j};
               if ischar(v)
                  fprintf(fid, ' %s', v);
               elseif isempty(v)
                  fprintf(fid, ' <empty>');
               else
                  fprintf(fid, ' %.10g', v);
               end
            end
            fprintf(fid, '\n');
         end
      end
   end
end
