import numpy as np
import pandas as pd
import lmfit

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from ipywidgets import widgets, Layout

from .load_data import load_scan

def fit_scan(scanno, xcol, ycol, normcol=None, visualize = False,
                  getTrace = False, getData=False):
    frame = load_scan(scanno)
    x_data = frame[xcol].values
    y_data = frame[ycol].values
    if normcol:
        norm_data = frame[normcol].values
        y_data = np.array(y_data)/np.array(norm_data)
    
    model1 = lmfit.models.PseudoVoigtModel()
    modelq = lmfit.models.LinearModel()
    model = model1+modelq

    pars1 = model1.guess(y_data -  np.min(y_data), x_data)
    pars_q = modelq.guess(np.array(y_data[:5] + y_data[-5:]), 
                            np.array(x_data[:5] + x_data[-5:]))
    pars = pars1 + pars_q
    
    output = model.fit(y_data, pars, x=x_data)

    if visualize or getTrace:
        trace = px.scatter(x=x_data, y=y_data)
        trace.layout.xaxis.title = xcol
        if type(ycol) == dict:
            trace.layout.yaxis.title = 'Intensity'
        else:
            trace.layout.yaxis.title = ycol
            
        xdatanew = np.linspace(x_data[0], x_data[-1], len(x_data)*10)
        ydatanew = model.eval(params=output.params, x=xdatanew)
            
        # fitTrace = px.line(x=xdata, y=ydata, markers=True)
        trace.add_trace(go.Scatter(x=xdatanew, y=ydatanew, 
                                    mode='lines', name='Fit'))
        
        if getTrace:
            return trace, output
        
        trace.show()
        
    if getData:
        return x_data, y_data, output
    
    return output

# def graph_run(scans, xcol, ycol, zcol=None, normcol=None):
        
#     # if type(ycol) == dict:
#     #     ycolName = 'Intensity'
#     # else:
#     #     ycolName = ycol
    
#     dataFrames = []
    
#     for scanno in scans:
        
#         xdata, ydata, output = fit_scan(scanno, xcol=xcol, ycol=ycol,
#                                                 normcol=normcol, getData=True)
        
#         baseline = load_scan(scanno, stream_name="baseline")
#         zval= baseline[zcol].mean()
#         zdata = zval*np.ones(len(ydata))
#         frame = pd.DataFrame(data={xcol: xdata, ycol: ydata, zcol:zdata}) 
#         dataFrames.append(frame)
        
        
#     dataFrame = pd.concat(dataFrames)
    
#     plasmaColors = pc.sample_colorscale(pc.sequential.Plasma,
#                             [n/(len(scans)-1) for n in range(len(scans))])
    
#     fig = px.line(dataFrame,x=xcol, y=ycol, color=zcol, markers=True,
#             color_discrete_sequence = plasmaColors)
        
    
#     fig.show()

# def analyze_run(scans, xcol, ycol, zcol, normcol=None, 
#                    paramsFrame=False):
        
#         # if type(ycol) == dict:
#         #     ycolName = 'Intensity'
#         # else:
#         #     ycolName = ycol
        
#         dataFrames = []
#         fitFrames = []
#         paramFrame = pd.DataFrame()
#         model = lmfit.models.PseudoVoigtModel() + lmfit.models.LinearModel()
        
#         for scanno in scans:
            
#             xdata, ydata, output = fit_scan(scanno, xcol=xcol, ycol=ycol,
#                                                   normcol=normcol, getData=True)
            

#             baseline = load_scan(scanno, stream_name="baseline")
#             zval= baseline[zcol].mean()
#             zdata = zval*np.ones(len(ydata))
#             frame = pd.DataFrame(data={xcol: xdata, ycol: ydata, zcol:zdata}) 
#             dataFrames.append(frame)
            
#             xdatanew = np.linspace(xdata[0], xdata[-1], len(xdata)*10)
#             ydatanew = model.eval(params=output.params, x=xdatanew)
#             zdatanew = zval*np.ones(len(ydatanew))
#             frameFit = pd.DataFrame(data={xcol: xdatanew, ycol + ' fit': ydatanew, 
#                                           zcol:zdatanew}) 
#             fitFrames.append(frameFit)
            
#             paramFrame.loc[scanno, zcol] = zval
#             for param in output.params:
#                 paramFrame.loc[scanno, param] = output.params[param].value
                
                
#         if paramsFrame:
#             return paramFrame
        
#         #Left panel with raw data and fit graph
            
#         dataFrame = pd.concat(dataFrames)
#         fitFrame = pd.concat(fitFrames)
        
#         plasmaColors = pc.sample_colorscale(pc.sequential.Plasma,
#                                 [n/(len(scans)-1) for n in range(len(scans))])
        
#         fig = px.line(dataFrame,x=xcol, y=ycol, color=zcol, markers=True,
#                 color_discrete_sequence = plasmaColors)
#         for data in fig.data:
#             data.mode = 'markers'
            
#         fig2 = px.line(fitFrame,x=xcol, y=ycol+' fit', color=zcol, 
#                 color_discrete_sequence = plasmaColors)
        
        
#         for i, data in enumerate(fig2.data):
#             fig.add_trace(data)
        
        
#         Fig = go.FigureWidget(data=fig)
        
        
#         Fig.update_layout(width=600, height=600)
        
        
#         #Right panel with widget and z dependence
        
#         textbox = widgets.Dropdown(
#         description='Fit param.: ',
#         value='center',
#         options=[i for i in paramFrame.columns.to_list() if i not in [zcol]]
#         )
        
#         trace = px.line(paramFrame, x=zcol, y='center', markers=True)
#         trace.data[0].line.color = 'green'
#         g = go.FigureWidget(data=trace)
        
#         g.update_layout(width=400, height=500)
        
#         def response(change):
            
#             paramYCol = textbox.value
#             with g.batch_update():
#                 g.data[0].y = paramFrame[paramYCol]
#                 g.layout.yaxis.title = paramYCol
                
#         textbox.observe(response, names='value')
        
#         container = widgets.VBox([textbox, g], 
#                                  layout = Layout(margin='60px 0 0 0'))
        
#         container2 = widgets.HBox([Fig, container])
        
        
#         return container2

def graph_run(scans, xcol, ycol, zcol='Scan', normcol=None):
        
    # if type(ycol) == dict:
    #     ycolName = 'Intensity'
    # else:
    #     ycolName = ycol
    
    dataFrames = []
    
    for scanno in scans:
        
        xdata, ydata, _ = fit_scan(scanno, xcol=xcol, ycol=ycol,
                                                normcol=normcol, getData=True)
        
        if  zcol=='Scan':
            zval = scanno
        else:
            baseline = load_scan(scanno, stream_name="baseline")
            zval= baseline[zcol].mean()
        zdata = zval*np.ones(len(ydata))
        frame = pd.DataFrame(data={xcol: xdata, ycol: ydata, zcol:zdata})
        frame.head()
        dataFrames.append(frame)

        
        
    dataFrame = pd.concat(dataFrames)

    
    plasmaColors = pc.sample_colorscale(pc.sequential.Plasma,
                            [n/(len(scans)-1) for n in range(len(scans))])
    
    fig = px.line(dataFrame,x=xcol, y=ycol, color=zcol, markers=True,
            color_discrete_sequence = plasmaColors)
        
    
    fig.show()

def analyze_run(scans, xcol, ycol, zcol='Scan', normcol=None, 
                   paramsFrame=False):
        
        # if type(ycol) == dict:
        #     ycolName = 'Intensity'
        # else:
        #     ycolName = ycol
        
        dataFrames = []
        fitFrames = []
        paramFrame = pd.DataFrame()
        model = lmfit.models.PseudoVoigtModel() + lmfit.models.LinearModel()
        
        for scanno in scans:
            
            xdata, ydata, output = fit_scan(scanno, xcol=xcol, ycol=ycol,
                                                  normcol=normcol, getData=True)
            

            baseline = load_scan(scanno, stream_name="baseline")
            if zcol=='Scan':
                zval = scanno
            else:
                baseline = load_scan(scanno, stream_name="baseline")
                zval= baseline[zcol].mean()
            zdata = zval*np.ones(len(ydata))
            frame = pd.DataFrame(data={xcol: xdata, ycol: ydata, zcol:zdata}) 
            dataFrames.append(frame)
            
            xdatanew = np.linspace(xdata[0], xdata[-1], len(xdata)*10)
            ydatanew = model.eval(params=output.params, x=xdatanew)
            zdatanew = zval*np.ones(len(ydatanew))
            frameFit = pd.DataFrame(data={xcol: xdatanew, ycol + ' fit': ydatanew, 
                                          zcol:zdatanew}) 
            fitFrames.append(frameFit)
            
            paramFrame.loc[scanno, zcol] = zval
            for param in output.params:
                paramFrame.loc[scanno, param] = output.params[param].value
                
                
        if paramsFrame:
            return paramFrame
        
        #Left panel with raw data and fit graph
            
        dataFrame = pd.concat(dataFrames)
        fitFrame = pd.concat(fitFrames)
        
        plasmaColors = pc.sample_colorscale(pc.sequential.Plasma,
                                [n/(len(scans)-1) for n in range(len(scans))])
        
        fig = px.line(dataFrame,x=xcol, y=ycol, color=zcol, markers=True,
                color_discrete_sequence = plasmaColors)
        for data in fig.data:
            data.mode = 'markers'
            
        fig2 = px.line(fitFrame,x=xcol, y=ycol+' fit', color=zcol, 
                color_discrete_sequence = plasmaColors)
        
        
        for i, data in enumerate(fig2.data):
            fig.add_trace(data)
        
        
        Fig = go.FigureWidget(data=fig)
        
        
        Fig.update_layout(width=600, height=600)
        
        
        #Right panel with widget and z dependence
        
        textbox = widgets.Dropdown(
        description='Fit param.: ',
        value='center',
        options=[i for i in paramFrame.columns.to_list() if i not in [zcol]]
        )
        
        trace = px.line(paramFrame, x=zcol, y='center', markers=True)
        trace.data[0].line.color = 'green'
        g = go.FigureWidget(data=trace)
        
        g.update_layout(width=400, height=500)
        
        def response(change):
            
            paramYCol = textbox.value
            with g.batch_update():
                g.data[0].y = paramFrame[paramYCol]
                g.layout.yaxis.title = paramYCol
                
        textbox.observe(response, names='value')
        
        container = widgets.VBox([textbox, g], 
                                 layout = Layout(margin='60px 0 0 0'))
        
        container2 = widgets.HBox([Fig, container])
        
        
        return container2