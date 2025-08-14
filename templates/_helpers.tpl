{{/*
Return the fully qualified name of the chart.
This automatically adapts to whatever .Chart.Name is at render time.
*/}}
{{- define (print .Chart.Name ".fullname") -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Return the short name of the chart.
*/}}
{{- define (print .Chart.Name ".name") -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}