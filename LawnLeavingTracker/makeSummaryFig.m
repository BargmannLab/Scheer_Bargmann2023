function makeSummaryFig(SUMMARY_STRUCT,EXIT_STRUCT,POKE_STRUCT,stat_int,titlestr)
%makeSummaryFig.m This function generates a summary figure from a
%summary_struct
close all; figure();
timenow = datestr(now,'mm_dd_yy');
tracklen = stat_int(2)-stat_int(1);
OK_frames_inlawn = SUMMARY_STRUCT.OK_FRAMES_IN_LAWN;

timevec = 0:3*60*5:tracklen;
timevec_labels = string(num2cell((timevec./180)+20));
[ha, ~] = tight_subplot(5, 1, 0.05, 0.05, 0.05); %ha is a vector of axes
set(gcf,'Position',[200 200 1500 800]); set(gcf,'Color','w');

%1. Plot Lawn Entries and Exits
axes(ha(1)); hold on;
xlim([timevec(1) timevec(end)]);
xticks(timevec);
xticklabels(timevec_labels);
% h_il = scatter(OK_frames_inlawn-stat_int(1),0.5*ones(size(OK_frames_inlawn)),5,'b','filled'); %plot the intervals when the worm is in the lawn
CIL = find(SUMMARY_STRUCT.CENTROIDINLAWN);
h_il = scatter(CIL,0.5*ones(size(CIL)),5,'b','filled'); %plot the intervals when the worm is in the lawn
if ~isempty(EXIT_STRUCT.EXITS_DURING_INTERVAL)
    h_ex = vline(EXIT_STRUCT.EXITS_DURING_INTERVAL,'r');
end
if ~isempty(EXIT_STRUCT.ENTERS_DURING_INTERVAL)
    h_en = vline(EXIT_STRUCT.ENTERS_DURING_INTERVAL,'g');
end
str = ['   Exit Rate = ' num2str(round(EXIT_STRUCT.EXIT_RATE_STATIC,2))];
text(0.05,1.5,str);
[Xf, Yf] = ds2nfu(50, 2);
str = ['   Frac. Time In Lawn = ' num2str(round(length(OK_frames_inlawn)/tracklen,2))];
text(0.05,1,str);

if exist('h_ex','var') && exist('h_en','var')
    legend([h_il(1),h_ex(1),h_en(1)],{'centroid in lawn','lawn EXIT','lawn ENTRY'},'Box','off');
elseif ~exist('h_ex','var') && exist('h_en','var')
    legend([h_il(1),h_en(1)],{'centroid in lawn','lawn ENTRY'},'Box','off');
elseif exist('h_ex','var') && ~exist('h_en','var')
    legend([h_il(1),h_ex(1)],{'centroid in lawn','lawn EXIT'},'Box','off');
else
    legend(h_il(1),{'centroid in lawn'},'Box','off');
end

ylabel('lawn leaving events / minute','FontSize',9);

%2. Plot Headpokes (Fwd + Rev + Pause)
axes(ha(2)); hold on;
xlim([timevec(1) timevec(end)]);
xticks(timevec);
xticklabels(timevec_labels);
if ~isempty(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_FWD)
    h_f = vline(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_FWD,'g');
end
if ~isempty(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_REV)
    h_r = vline(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_REV,'m');
end
if ~isempty(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_PAUSE)
    h_p = vline(POKE_STRUCT.HEADPOKES_DURING_INTERVAL_PAUSE,'b');
end
[Xf, Yf] = ds2nfu(50, 2);
str = ['   Fwd Poke Rate = ' num2str(round(POKE_STRUCT.POKE_RATE_STATIC_FWD,2))];
text(Xf,Yf,str);
[Xf, Yf] = ds2nfu(50, 1);
str = ['   Rev Poke Rate = ' num2str(round(POKE_STRUCT.POKE_RATE_STATIC_REV,2))];
text(Xf,Yf,str);
[Xf, Yf] = ds2nfu(50, 0);
str = ['   Pause Poke Rate = ' num2str(round(POKE_STRUCT.POKE_RATE_STATIC_PAUSE,2))];
text(Xf,Yf,str);
ylabel('head poke events / minute','FontSize',9);

%3. Plot Speed
axes(ha(3)); hold on;
xlim([timevec(1) timevec(end)]);
xticks(timevec);
xticklabels(timevec_labels);
plot(1:length(SUMMARY_STRUCT.SPEED),SUMMARY_STRUCT.SPEED,'Color',[43 109 242]./255,'LineWidth',1);
speedvec = -0.3:0.1:0.3;
ylim([speedvec(1) speedvec(end)])
yticks(speedvec);
speed_labels = string(num2cell(speedvec));
yticklabels(speed_labels);
str1 = ['   Speed ON  food = ' num2str(round(SUMMARY_STRUCT.MEAN_SPEED_ON_FOOD,2))];
text(50,-0.09,str1);
str2 = ['   Speed OFF food = ' num2str(round(SUMMARY_STRUCT.MEAN_SPEED_OFF_FOOD,2))];
text(50,-0.2,str2);
ylabel('Speed (mm/sec)','FontSize',9);

%4. Plot AngSpeed
axes(ha(4)); hold on;
xlim([timevec(1) timevec(end)]);
xticks(timevec);
xticklabels(timevec_labels);
plot(1:length(SUMMARY_STRUCT.ANGSPEED),SUMMARY_STRUCT.ANGSPEED,'Color',[239 141 55]./255,'LineWidth',1);
angvec = -180:90:180;
ylim([angvec(1) angvec(end)]);
yticks(angvec);
angvec_labels = string(num2cell(angvec));
yticklabels(angvec_labels);
ylabel('Angular Speed (deg/sec)','FontSize',9);

%5. Plot Roaming and Dwelling (2D + HMM)
axes(ha(5)); hold on;
xlim([timevec(1) timevec(end)]);
xticks(timevec);
xticklabels(timevec_labels);
plot(1:length(SUMMARY_STRUCT.ROAMDWELL_2D), SUMMARY_STRUCT.ROAMDWELL_2D);
plot(1:length(SUMMARY_STRUCT.ROAMDWELL_HMM), SUMMARY_STRUCT.ROAMDWELL_HMM,'LineWidth',2);
str = ['Frac. Time Spent Roaming = ' num2str(round(SUMMARY_STRUCT.FRAC_ROAMING_HMM,2))];
text(75,1.85,str);
ylabel('Dwelling <---> Roaming','FontSize',9);
xlabel('time (min)','FontSize',9);

linkaxes(ha,'x');
axes(ha(1));
title(titlestr, 'Interpreter','none');
saveas(gcf, [titlestr '_SUMMARY_' timenow '.fig']);
close all;

end

